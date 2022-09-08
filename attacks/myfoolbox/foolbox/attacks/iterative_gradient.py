import numpy as np
from collections import Iterable
import abc
import logging
import cv2
import random
import torch.nn.functional as F
import torch
from torch.autograd import Variable

from .base import Attack
from .base import generator_decorator


class IterativeGradientBaseAttack(Attack):
    """Common base class for iterative gradient attacks."""

    @abc.abstractmethod
    def _gradient(self, a, x):
        raise NotImplementedError

    def _run(self, a, epsilons, max_epsilon, steps,
             confidence                       ###########################
             ):
        logging.warning(
            "Please consider using the L2BasicIterativeAttack,"
            " the LinfinityBasicIterativeAttack or one of its"
            " other variants such as the ProjectedGradientDescent"
            " attack."
        )
        if not a.has_gradient():
            logging.fatal(
                "Applied gradient-based attack to model that "
                "does not provide gradients."
            )
            return

        x = a.unperturbed
        min_, max_ = a.bounds()

        if not isinstance(epsilons, Iterable):
            assert isinstance(epsilons, int)
            max_epsilon_iter = max_epsilon / steps
            epsilons = np.linspace(0, max_epsilon_iter, num=epsilons + 1)[1:]

        for epsilon in epsilons:
            perturbed = x

            for st in range(100):
                gradient = yield from self._gradient(a, perturbed)

                perturbed = perturbed + gradient * epsilon
                perturbed = np.clip(perturbed, min_, max_)

                # yield from a.forward_one(perturbed)

                yield from a.forward_one3(perturbed, confidence=confidence, cmin=a.original_class,
                                                          cmax=a.target_class)

                # we don't return early if an adversarial was found
                # because there might be a different epsilon
                # and/or step that results in a better adversarial


class IterativeGradientAttack(IterativeGradientBaseAttack):
    """Like GradientAttack but with several steps for each epsilon.

    """

    @generator_decorator
    def as_generator(self, a, epsilons=100, max_epsilon=1, steps=10):

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

        yield from self._run(a, epsilons=epsilons, max_epsilon=max_epsilon, steps=steps)

    def _gradient(self, a, x):
        min_, max_ = a.bounds()
        gradient = yield from a.gradient_one(x)
        gradient_norm = np.sqrt(np.mean(np.square(gradient)))
        gradient = gradient / (gradient_norm + 1e-8) * (max_ - min_)
        return gradient


class IterativeGradientSignAttack(IterativeGradientBaseAttack):
    """Like GradientSignAttack but with several steps for each epsilon.

    """

    @generator_decorator
    def as_generator(self, a, epsilons=100, max_epsilon=1, steps=10,
                     confidence=0                  #################
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

        yield from self._run(a, epsilons=epsilons, max_epsilon=max_epsilon, steps=steps,
                             confidence=confidence            #########################
                             )

    def _gradient(self, a, x):
        min_, max_ = a.bounds()
        gradient = yield from a.gradient_one(x)
        gradient = np.sign(gradient) * (max_ - min_)
        return gradient


class Momentum_on_IFGSM(IterativeGradientBaseAttack):
    """Like GradientSignAttack but with several steps for each epsilon.

    """

    @generator_decorator
    def as_generator(self, a, epsilons=100, max_epsilon=1, steps=10, decay_factor=1.0,
                     confidence=0                  #################
                     ):

        self._decay_factor = decay_factor

        yield from self._run(a, epsilons=epsilons, max_epsilon=max_epsilon, steps=steps,
                             confidence=confidence,            #########################
                             )

    def _gradient(self, a, x, strict=True, gradient_args={}):
        # get current gradient
        gradient = yield from a.gradient_one(x, strict=strict)
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
        success = yield from super(Momentum_on_IFGSM, self)._run(
            *args, **kwargs)
        return success


class IterativeGradientBaseAttack2(Attack):
    """Common base class for iterative gradient attacks."""

    @abc.abstractmethod
    def _gradient(self, a, x):
        raise NotImplementedError

    def _input_diversity(self, _image, div_prob, low=270, high=299):
        image = _image[np.newaxis, :, :, :]
        image = Variable(torch.from_numpy(image))
        if random.random() > div_prob:
            return _image
        rnd = random.randint(low, high)
        rescaled = F.interpolate(image, size=[rnd, rnd], mode='nearest')
        h_rem = high - rnd
        w_rem = high - rnd
        pad_top = random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
        out = padded.squeeze().numpy()
        return out

    def _head_input_diversity(self, _image, div_prob):
        if _image.ndim > 3:
            out = np.zeros(_image.shape, dtype=np.float32)
            for t in range(_image.shape[1]):
                sub_img = _image[:,t,:,:]
                out[:,t,:,:] = self._input_diversity(sub_img, div_prob=div_prob)

        else:
            out = self._input_diversity(_image, div_prob=div_prob)
        return out

    def _run(self, a, epsilons, max_epsilon, steps,
             confidence, div_prob                       ###########################
             ):
        logging.warning(
            "Please consider using the L2BasicIterativeAttack,"
            " the LinfinityBasicIterativeAttack or one of its"
            " other variants such as the ProjectedGradientDescent"
            " attack."
        )
        if not a.has_gradient():
            logging.fatal(
                "Applied gradient-based attack to model that "
                "does not provide gradients."
            )
            return

        x = a.unperturbed
        min_, max_ = a.bounds()

        if not isinstance(epsilons, Iterable):
            assert isinstance(epsilons, int)
            max_epsilon_iter = max_epsilon / steps
            epsilons = np.linspace(0, max_epsilon_iter, num=epsilons + 1)[1:]
            # epsilon = epsilons

        for epsilon in epsilons:
            perturbed = x
            # epsilon = 0.002

            for _ in range(steps):
                _perturbed = self._head_input_diversity(perturbed, div_prob=div_prob)   ##########
                gradient = yield from self._gradient(a, _perturbed)    ##########

                perturbed = perturbed + gradient * epsilon
                perturbed = np.clip(perturbed, min_, max_)

                # yield from a.forward_one(perturbed)

            yield from a.forward_one3(perturbed, confidence=confidence, cmin=a.original_class,
                                                          cmax=a.target_class)

                # we don't return early if an adversarial was found
                # because there might be a different epsilon
                # and/or step that results in a better adversarial


class Momentum_Diversity_IFGSM(IterativeGradientBaseAttack2):
    """Like GradientSignAttack but with several steps for each epsilon.

    """

    @generator_decorator
    def as_generator(self, a, epsilons=100, max_epsilon=1, steps=10, decay_factor=1.0,
                     confidence=0, div_prob=0.5                  #################
                     ):

        self._decay_factor = decay_factor

        yield from self._run(a, epsilons=epsilons, max_epsilon=max_epsilon, steps=steps,
                             confidence=confidence, div_prob=div_prob            #########################
                             )

    def _gradient(self, a, x, strict=True, gradient_args={}):
        # get current gradient
        gradient = yield from a.gradient_one(x, strict=strict)
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
        success = yield from super(Momentum_Diversity_IFGSM, self)._run(
            *args, **kwargs)
        return success



