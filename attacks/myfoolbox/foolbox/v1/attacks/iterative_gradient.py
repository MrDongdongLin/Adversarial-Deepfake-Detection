import numpy as np
from collections import Iterable
import abc
import logging
import cv2

from .base import Attack
from .base import call_decorator


class IterativeGradientBaseAttack(Attack):
    """Common base class for iterative gradient attacks."""

    @abc.abstractmethod
    def _gradient(self, a, x):
        raise NotImplementedError

    def _run(self, a, epsilons, max_epsilon, steps, input_D,
             confidence,
             ):
        logging.warning(
            "Please consider using the L2BasicIterativeAttack,"
            " the LinfinityBasicIterativeAttack or one of its"
            " other variants such as the ProjectedGradientDescent"
            " attack."
        )
        if not a.has_gradient():
            return

        x = a.unperturbed
        min_, max_ = a.bounds()

        if not isinstance(epsilons, Iterable):
            assert isinstance(epsilons, int)
            max_epsilon_iter = max_epsilon / steps
            epsilons = np.linspace(0, max_epsilon_iter, num=epsilons + 1)[1:]

        # s_record = 100
        for epsilon in epsilons:
            perturbed = x

            for cur_step in range(steps):
                ############## wenjie 5/22
                if input_D is True:
                    perturbed = self._input_diversity(perturbed)
                #################################################################################
                gradient = self._gradient(a, perturbed)

                perturbed = perturbed + gradient * epsilon
                perturbed = np.clip(perturbed, min_, max_)

                # a.forward_one(perturbed)

                _, _, s_best = a.forward_one3_record_steps(perturbed, confidence=confidence, cmin=a.original_class,
                               cmax=a.target_class, s=cur_step)

                if s_best != 100:
                    s_record = s_best

        with open(r'/media/hdddati2/wenjie/Input_Diversity/s_confs12.5_VGG_median.txt', 'a') as f:
            f.write('s_record={}\n'.format(s_record))


                # we don't return early if an adversarial was found
                # because there might be a different epsilon
                # and/or step that results in a better adversarial


class IterativeGradientAttack(IterativeGradientBaseAttack):
    """Like GradientAttack but with several steps for each epsilon.

    """

    @call_decorator
    def __call__(
        self,
        input_or_adv,
        label=None,
        unpack=True,
        epsilons=100,
        max_epsilon=1,
        steps=10,
        input_D = False,
        confidence=0,
    ):

        """Like GradientAttack but with several steps for each epsilon.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        epsilons : int or Iterable[float]
            Either Iterable of step sizes in the gradient direction
            or number of step sizes between 0 and max_epsilon that should
            be tried.
        max_epsilon : float
            Largest step size if epsilons is not an iterable.
        steps : int
            Number of iterations to run.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        self._run(a, epsilons=epsilons, max_epsilon=max_epsilon, steps=steps, input_D=False, confidence=confidence)

    def _gradient(self, a, x):
        min_, max_ = a.bounds()
        gradient = a.gradient_one(x)
        gradient_norm = np.sqrt(np.mean(np.square(gradient)))
        gradient = gradient / (gradient_norm + 1e-8) * (max_ - min_)
        return gradient


class IterativeGradientSignAttack(IterativeGradientBaseAttack):
    """Like GradientSignAttack but with several steps for each epsilon.

    """

    @call_decorator
    def __call__(
        self,
        input_or_adv,
        label=None,
        unpack=True,
        epsilons=100,
        max_epsilon=1,
        steps=10,
        input_D=False,
        confidence=0,
    ):

        """Like GradientSignAttack but with several steps for each epsilon.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        epsilons : int or Iterable[float]
            Either Iterable of step sizes in the direction of the sign of
            the gradient or number of step sizes between 0 and max_epsilon
            that should be tried.
        max_epsilon : float
            Largest step size if epsilons is not an iterable.
        steps : int
            Number of iterations to run.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        self._run(a, epsilons=epsilons, max_epsilon=max_epsilon, steps=steps, input_D=False, confidence=confidence)

    def _gradient(self, a, x):
        min_, max_ = a.bounds()
        gradient = a.gradient_one(x)
        gradient = np.sign(gradient) * (max_ - min_)
        return gradient

########################################################################################################################
class Momentum_on_IFGSM(IterativeGradientBaseAttack):
    """Like GradientSignAttack but with several steps for each epsilon.

    """

    @call_decorator
    def __call__(
        self,
        input_or_adv,
        label=None,
        unpack=True,
        epsilons=100,
        max_epsilon=1,
        steps=10,
        input_D=False,
        decay_factor=1.0,
        confidence=0,
                ):

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        self._decay_factor = decay_factor

        self._run(a, epsilons=epsilons, max_epsilon=max_epsilon, steps=steps, input_D=False,
                 confidence=confidence,            #########################
                 )

    def _gradient(self, a, x, strict=True, gradient_args={}):
        # get current gradient
        gradient = a.gradient_one(x, strict=strict)
        gradient = gradient / max(1e-12, np.mean(np.abs(gradient)))

        # combine with history of gradient as new history
        self._momentum_history = self._decay_factor * self._momentum_history + gradient

        # use history
        gradient = self._momentum_history
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient

    def _run(self, *args, **kwargs):
        # reset momentum history every time we restart
        # gradient descent
        self._momentum_history = 0
        success = super(Momentum_on_IFGSM, self)._run(
            *args, **kwargs)
        return success

############################### wenjie 5/22 ############################################################################
class DI_on_IFGSM(IterativeGradientBaseAttack):
    """Like GradientSignAttack but with several steps for each epsilon.

    """

    @call_decorator
    def __call__(
        self,
        input_or_adv,
        label=None,
        unpack=True,
        epsilons=100,
        max_epsilon=1,
        steps=10,
        input_D=True,
        decay_factor=0.0,
        confidence=0,
                ):

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        self._decay_factor = decay_factor

        self._run(a, epsilons=epsilons, max_epsilon=max_epsilon, steps=steps, input_D=True,
                 confidence=confidence,            #########################
                 )

    def _input_diversity(self, x):
        image_resize = 100
        image_width = 128
        prob = 0.5
        rnd = np.random.randint(image_resize, image_width)
        rescaled = cv2.resize(x, (rnd, rnd), interpolation=cv2.INTER_NEAREST)
        h_rem = image_width - rnd
        w_rem = image_width - rnd
        pad_top = np.random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        padded = np.pad(rescaled, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=(0, 0))
        rnd_prob = np.random.rand()
        if rnd_prob < prob:
            return padded.reshape([image_width, image_width, 1])
        else:
            return x

    def _gradient(self, a, x, strict=True, gradient_args={}):
        # get current gradient
        gradient = a.gradient_one(x, strict=strict)
        gradient = gradient / max(1e-12, np.mean(np.abs(gradient)))

        # combine with history of gradient as new history
        self._momentum_history = self._decay_factor * self._momentum_history + gradient

        # use history
        gradient = self._momentum_history
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient

    def _run(self, *args, **kwargs):
        # reset momentum history every time we restart
        # gradient descent
        self._momentum_history = 0
        success = super(DI_on_IFGSM, self)._run(
            *args, **kwargs)
        return success