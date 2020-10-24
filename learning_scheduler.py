import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.PolynomialDecay):
    
    def __init__(self,
      initial_learning_rate,
      decay_steps,
      end_learning_rate=0.0001,
      power=1.0,
      cycle=False,
      name=None,
      num_warmup_steps=1000):
        
        # Since we have a custom __call__() method, we pass cycle=False when calling `super().__init__()` and
        # in self.__call__(), we simply do `step = step % self.decay_steps` to have cyclic behavior.
        super(CustomSchedule, self).__init__(initial_learning_rate, decay_steps, end_learning_rate, power, cycle=False, name=name)
        
        self.num_warmup_steps = num_warmup_steps
        
        self.cycle = tf.constant(cycle, dtype=tf.bool)
        
    def __call__(self, step):
        """ `step` is actually the step index, starting at 0.
        """
        
        # For cyclic behavior
        step = tf.cond(self.cycle and step >= self.decay_steps, lambda: step % self.decay_steps, lambda: step)
        
        learning_rate = super(CustomSchedule, self).__call__(step)

        # Copy (including the comments) from original bert optimizer with minor change.
        # Ref: https://github.com/google-research/bert/blob/master/optimization.py#L25
        
        # Implements linear warmup: if global_step < num_warmup_steps, the
        # learning rate will be `global_step / num_warmup_steps * init_lr`.
        if self.num_warmup_steps > 0:
            
            steps_int = tf.cast(step, tf.int32)
            warmup_steps_int = tf.constant(self.num_warmup_steps, dtype=tf.int32)

            steps_float = tf.cast(steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            # The first training step has index (`step`) 0.
            # The original code use `steps_float / warmup_steps_float`, which gives `warmup_percent_done` being 0,
            # and causing `learning_rate` = 0, which is undesired.
            # For this reason, we use `(steps_float + 1) / warmup_steps_float`.
            # At `step = warmup_steps_float - 1`, i.e , at the `warmup_steps_float`-th step, 
            #`learning_rate` is `self.initial_learning_rate`.
            warmup_percent_done = (steps_float + 1) / warmup_steps_float
            
            warmup_learning_rate = self.initial_learning_rate * warmup_percent_done

            is_warmup = tf.cast(steps_int < warmup_steps_int, tf.float32)
            learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
                        
        return learning_rate
    