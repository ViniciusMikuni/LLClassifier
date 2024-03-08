import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
import utils
from tensorflow.keras.losses import mse, mae
import tensorflow_probability as tfp
from architecture import DeepSetsAtt, Resnet
import gc
from tensorflow_probability.python.math.diag_jacobian import diag_jacobian
#tf.random.set_seed(1234)

class GSGM(keras.Model):
    """Score based generative model"""
    def __init__(self,
                 num_feat,
                 num_jet,
                 num_embed = 64,
                 projection_dim = 64,
                 num_layer = 4,
                 num_head = 2,
                 num_part=50,name='GSGM'):
        super(GSGM, self).__init__()

        
        #self.activation = layers.LeakyReLU(alpha=0.01)
        self.num_feat = num_feat
        self.num_jet = num_jet
        
        self.num_embed = num_embed
        self.max_part = num_part        
        self.ema=0.999
        self.shape = (-1,1,1)
        self.num_steps = 256
        
        self.loss_tracker = keras.metrics.Mean(name="loss")


        #Transformation applied to conditional inputs
        inputs_time = Input((1),name='input_time')
        inputs_jet = Input((self.num_jet),name='input_jets')
        inputs_particles = Input((None,self.num_feat),name='input_particles')
        inputs_mask = Input((None,1),name='input_mask') #mask to identify zero-padded objects

        
        outputs = DeepSetsAtt(
            self.FF(inputs_particles),
            inputs_time,
            inputs_jet,
            inputs_mask,
            num_feat=self.num_feat,
            num_heads= num_head,
            num_transformer = num_layer,
            projection_dim = projection_dim,
        )

        self.model_part = keras.Model(inputs=[inputs_particles,inputs_jet,inputs_time,inputs_mask],
                                      outputs=outputs)
        
                   
        outputs = Resnet(
            inputs_jet,
            inputs_time,
            self.num_jet,
            num_layer = 3,
            projection_dim= 2*projection_dim,
        )
        
        self.model_jet = keras.Model(inputs=[inputs_jet,inputs_time],
                                     outputs=outputs)


        self.ema_jet = keras.models.clone_model(self.model_jet)
        self.ema_part = keras.models.clone_model(self.model_part)
        
        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]
    

    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions,dtype=tf.float32)
    
    def FF(self,features,expand=False):
        #Gaussian features to the inputs
        max_proj = 8
        min_proj = 6
        freq = tf.range(start=min_proj, limit=max_proj, dtype=tf.float32)
        freq = 2.**(freq) * 2 * np.pi        
        x = features
        freq = tf.tile(freq[None, :], ( 1, tf.shape(x)[-1]))  
        h = tf.repeat(x, max_proj-min_proj, axis=-1)
        if expand:
            angle = h*freq[None,:]
        else:
            angle = h*freq
        h = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        return tf.concat([features,h],-1)

    
    #@tf.function
    def get_logsnr_alpha_sigma(self,time,shape=None,eps=1e-9):
        #logsnr = self.logsnr_schedule_cosine(time,logsnr_min=self.minlogsnr, logsnr_max=self.maxlogsnr)
        alpha = tf.sqrt(1.0- tf.clip_by_value(time,eps,1.0))
        sigma = tf.sqrt(tf.clip_by_value(time,eps,1.0))
        logsnr = 2*tf.math.log(alpha/sigma)
        if shape is not None:
            logsnr = tf.reshape(tf.cast(logsnr,tf.float32),shape)
            alpha = tf.reshape(tf.cast(alpha,tf.float32),shape)
            sigma = tf.reshape(tf.cast(sigma,tf.float32),shape)
        return logsnr, alpha, sigma
        

    @tf.function
    def eval_model(self,model,x,t,jet=None,mask=None):
        if jet is None:
            score = model([x, t])
        else:
            score = model([x*mask,jet,t,mask])*mask
        return score


    @tf.function
    def get_score(self,model,x,t,jet=None,mask=None,const_shape=None):
        v = self.eval_model(model,x,t,jet,mask)
        _, alpha, sigma = self.get_logsnr_alpha_sigma(t,shape=const_shape)
        score = x - v*alpha/sigma
        return score

    
    @tf.function
    def train_step(self, inputs):

        batch_size = tf.shape(inputs['input_jets'])[0]
        random_t = tf.random.uniform((batch_size,1))
                    
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)
        
        with tf.GradientTape() as tape:
            #part
            eps = tf.random.normal((tf.shape(inputs['input_particles'])),
                                 dtype=tf.float32)*inputs['input_mask'][:,:,None]
            
            perturbed_x = alpha[:,None]*inputs['input_particles'] + eps * sigma[:,None]
            
            v_pred_part = self.model_part([
                perturbed_x,inputs['input_jets'],
                random_t,inputs['input_mask']])
            v_pred_part = tf.reshape(v_pred_part,(tf.shape(v_pred_part)[0], -1))
            
            v = alpha[:,None] * eps - sigma[:,None] * inputs['input_particles']
            v = tf.reshape(v,(tf.shape(v)[0], -1))
            loss_part = tf.reduce_sum(tf.square(v-v_pred_part))/(3*tf.reduce_sum(inputs['input_mask']))

                            
            #jet
            eps = tf.random.normal((tf.shape(inputs['input_jets'])),dtype=tf.float32)
            
            perturbed_x = alpha*inputs['input_jets'] + eps * sigma
            
            v_pred_jet = self.model_jet([perturbed_x,random_t])
            v = alpha* eps - sigma* inputs['input_jets']
            loss_jet = mse(v_pred_jet,v)

            loss = loss_part + loss_jet
 
        trainable_variables = self.model_jet.trainable_variables + self.model_part.trainable_variables
        self.optimizer.minimize(loss,trainable_variables,tape=tape)
        self.loss_tracker.update_state(loss)

            
        for weight, ema_weight in zip(self.model_jet.weights, self.ema_jet.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        for weight, ema_weight in zip(self.model_part.weights, self.ema_part.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)


        return {
            "loss": self.loss_tracker.result(), 
            "loss_part":tf.reduce_mean(loss_part),
            "loss_jet":tf.reduce_mean(loss_jet),
        }


    @tf.function
    def test_step(self, inputs):
        batch_size = tf.shape(inputs['input_jets'])[0]
        random_t = tf.random.uniform((batch_size,1))
                    
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)


        #part
        eps = tf.random.normal((tf.shape(inputs['input_particles'])),
                               dtype=tf.float32)*inputs['input_mask'][:,:,None]
        
        perturbed_x = alpha[:,None]*inputs['input_particles'] + eps * sigma[:,None]
            
        v_pred_part = self.model_part([
            perturbed_x,inputs['input_jets'],
            random_t,inputs['input_mask']])
        v_pred_part = tf.reshape(v_pred_part,(tf.shape(v_pred_part)[0], -1))
        
        v = alpha[:,None] * eps - sigma[:,None] * inputs['input_particles']
        v = tf.reshape(v,(tf.shape(v)[0], -1))
        loss_part = tf.reduce_sum(tf.square(v-v_pred_part))/(3*tf.reduce_sum(inputs['input_mask']))

        
        #jet
        eps = tf.random.normal((tf.shape(inputs['input_jets'])),dtype=tf.float32)
        
        perturbed_x = alpha*inputs['input_jets'] + eps * sigma
        
        v_pred_jet = self.model_jet([perturbed_x,random_t])
        v = alpha* eps - sigma* inputs['input_jets']
        loss_jet = mse(v_pred_jet,v)
        
        loss = loss_part + loss_jet
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(), 
            "loss_part":tf.reduce_mean(loss_part),
            "loss_jet":tf.reduce_mean(loss_jet),
        }

            
    @tf.function
    def call(self,x):        
        return self.model(x)


    def get_likelihood(self,part,jet,mask):
        start = time.time()

        ll_jet,normal_jet = self.Likelihood(jet,self.ema_jet,
                                 data_shape=[jet.shape[0],self.num_jet],
                                 const_shape=[-1,1],                                 
                                 )
        end = time.time()
        
        print("Time for calculating the likelihood of {} events is {} seconds".format(jet.shape[0],end - start))
        start = time.time()
        ll_part,normal_data = self.Likelihood(part,self.ema_part,
                                  data_shape=[part.shape[0],self.max_part,self.num_feat],
                                  const_shape = self.shape,
                                  jet=jet,mask=mask,)

        end = time.time()
        print("Time for calculating the likelihood of {} events is {} seconds".format(jet.shape[0],end - start))
        return ll_part, ll_jet,normal_data*mask,normal_jet


    def generate(self,nevts):
        start = time.time()
        jet_info,normal_jet = self.ODESampler(nevts,self.ema_jet,
                                   data_shape=[nevts,self.num_jet],
                                   const_shape=[-1,1]
                                   )
        
        end = time.time()
        print("Time for sampling {} events is {} seconds".format(nevts,end - start))

        nparts = np.expand_dims(np.clip(utils.revert_npart(jet_info[:,-1]),
                                        2,self.max_part),-1)
        
    
        mask = np.expand_dims(
            np.tile(np.arange(self.max_part),(nparts.shape[0],1)) < np.tile(nparts,(1,self.max_part)),-1)

        start = time.time()
        parts,normal_data = self.ODESampler(nevts,
                                self.ema_part,
                                data_shape=[nevts,self.max_part,self.num_feat],
                                const_shape = self.shape,
                                jet=jet_info,
                                mask=tf.convert_to_tensor(mask, dtype=tf.float32))
        end = time.time()
        print("Time for sampling {} events is {} seconds".format(nevts,end - start))
        return parts.numpy()*mask,jet_info.numpy(), normal_data*mask,normal_jet

    
    def get_sde(self,time,shape=None):
        logsnr,alpha,sigma = self.get_logsnr_alpha_sigma(time)

        f = 1.0/(2*alpha**2)
        g2 = 1.0/(alpha**2)
        
        if shape is None:
            shape=self.shape
        f = tf.reshape(f,shape)
        g2 = tf.reshape(g2,shape)
        return tf.cast(f,tf.float32), tf.cast(g2,tf.float32)


    def ODESampler(self,nevts,
                   model,
                   data_shape=None,
                   const_shape=None,
                   jet=None,
                   mask=None,
                   atol=1e-5,
                   eps=1e-5):

        from scipy import integrate

        init_x = self.prior_sde(data_shape)        
        shape = init_x.shape
        
        @tf.function
        def score_eval_wrapper(sample, time_steps,jet=None,mask=None):
            """A wrapper of the score-based model for use by the ODE solver."""
            sample = tf.cast(tf.reshape(sample,shape),tf.float32)
            time_steps = tf.reshape(time_steps,(sample.shape[0], 1))
            score = self.get_score(model,sample,time_steps,jet,mask,const_shape)
            return tf.reshape(score,[-1])


        def ode_func(t, x):        
            """The ODE function for use by the ODE solver."""
            time_steps = np.ones((shape[0],)) * t    
            f,g2 = self.get_sde(t,shape= (-1))
            return  f*x - 0.5*g2 * score_eval_wrapper(x, time_steps,jet,mask).numpy()
        
        res = integrate.solve_ivp(
            ode_func, (1.0-eps, eps), tf.reshape(init_x,[-1]).numpy(),
            rtol=atol, atol=atol, method='RK23')  
        print(f"Number of function evaluations: {res.nfev}")
        sample = tf.reshape(res.y[:, -1],shape)
        return sample, init_x


    def Likelihood(self,
                   sample,
                   model,
                   data_shape = None,
                   const_shape = None,
                   jet=None,
                   mask=None,
                   atol=1e-5,
                   rtol =5e-4,
                   eps = 1e-5,
                   exact = True,
    ):

        from scipy import integrate
        gc.collect()
        
        batch_size = sample.shape[0]        
        shape = sample.shape
        
        if mask is None:
            N = np.prod(shape[1:])
        else:
            N = np.sum(self.num_feat*mask,(1,2))

            
        def prior_likelihood(z):
            """The likelihood of a Gaussian distribution with mean zero and 
            standard deviation sigma."""
            shape = z.shape            
            return -N / 2. * np.log(2*np.pi) - np.sum(z.reshape((shape[0],-1))**2, -1) / 2. 

        
        @tf.function
        def divergence_eval_wrapper(sample, time_steps,
                                    jet=None,mask=None):
            
            sample = tf.cast(tf.reshape(sample,shape),tf.float32)
            time_steps = tf.reshape(time_steps,(sample.shape[0], 1))            
            f,g2 = self.get_sde(time_steps,shape= const_shape)
            
            epsilons = tfp.random.rademacher(sample.shape,dtype=tf.float32)
            if mask is not None:
                sample*=mask
                epsilons*=mask

            if exact:
                # Exact trace estimation
                fn = lambda x: f*x - 0.5*g2*self.get_score(model,x,time_steps,jet,mask,const_shape)
                pred, diag_jac = diag_jacobian(
                xs=sample, fn=fn, sample_shape=[batch_size])

                if isinstance(pred, list):
                    pred = pred[0]
                    if isinstance(diag_jac, list):
                        diag_jac = diag_jac[0]
            
                return tf.reshape(pred,[-1]), - tf.reduce_sum(tf.reshape(diag_jac,(batch_size,-1)), -1)
            else:                
                with tf.GradientTape(persistent=False,
                                     watch_accessed_variables=False) as tape:
                    tape.watch(sample)
                    score = self.get_score(model,sample,time_steps,jet,mask,const_shape)
                    drift = f*sample - 0.5*g2 *score
                
                jvp = tf.cast(tape.gradient(drift, sample,epsilons),tf.float32)            
                return  tf.reshape(drift,[-1]), - tf.reduce_sum(tf.reshape(jvp*epsilons,(batch_size,-1)), -1)



        def ode_func(t, x):        
            """The ODE function for use by the ODE solver."""
            time_steps = np.ones((batch_size,)) * t    
            sample = x[:-batch_size]
            logp = x[-batch_size:]
            sample_grad, logp_grad = divergence_eval_wrapper(sample, time_steps,jet,mask)
            return np.concatenate([sample_grad, logp_grad], axis=0)
    
        init_x = np.concatenate([sample.reshape([-1]),np.zeros((batch_size,))],0)
        res = integrate.solve_ivp(
            ode_func,
            (eps,1.0-eps),            
            init_x,t_eval=[1.0-eps],
            rtol=rtol, atol=atol, method='RK45')
        print(f"Number of function evaluations: {res.nfev}")
        zp = res.y[:, -1]
        z = zp[:-batch_size].reshape(shape)
        if mask is not None:
            z *= mask
            
        delta_logp = zp[-batch_size:].reshape(batch_size)
        prior_logp = prior_likelihood(z)
        return (prior_logp - delta_logp),z



    

    @tf.function
    def DDPMSampler(self,
                    nevts,
                    model,
                    data_shape=None,
                    const_shape=None,
                    jet=None,
                    mask=None):
        """Generate samples from score-based models with DDPM method.
        
        Args:
        cond: Conditional input
        model: Trained score model to use
        data_shape: Format of the data
        const_shape: Format for constants, should match the data_shape in dimensions
        jet: input jet conditional information if used
        mask: particle mask if used

        Returns: 
        Samples.
        """

        batch_size = nevts
        x = self.prior_sde(data_shape)

        for time_step in tf.range(self.num_steps, 0, delta=-1):
            t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / self.num_steps            
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t,shape=const_shape)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / self.num_steps,shape=const_shape)


            v = self.eval_model(model,x,t,jet,mask)
            mean = alpha * x - sigma * v
            eps = v * alpha + x * sigma                                        
            x = alpha_ * mean + sigma_ * eps
            
        return mean
