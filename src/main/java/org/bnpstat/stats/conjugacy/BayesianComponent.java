package org.bnpstat.stats.conjugacy;

/**
 * Created by hvhuynh on 11/6/2015.
 */

import java.io.Serializable;
import java.util.Objects;

/**
    This structure follows the most basic Bayesian setting:
        w ~  H(\lambda)
        x   ~  F(w)
        where:  \lambda is hyperparameter,
        H is the prior distribution
        w is the parameter
        F is the observation distribution, and
        x is the observation
        In many cases H and F are conjugate in the sense that posterior p(\mu | x, \lambda) is another distribution
        belong to the family of H. In general, we do not restrict ourselves to such conjugacy requirement.

    This is the interface to such a canonical Bayesian setting.
*/
public interface BayesianComponent extends Cloneable,Serializable {


    /**
     * Add a new observation into this model
     * @param observation - the observation to be added - it has to be cast into proper datatype
     *              depending on data we are working with
     */
    void add(Object observation)throws Exception;

    /**
     * Add a new observation into this model with a probability. This method is usually used with
     * variational inference methods
     * @param observation - the observation to be added - it has to be cast into proper datatype
     *              depending on data we are working with
     * @param prob - the probability observation belong to this component
     */
    void add(Object observation, double prob)throws Exception;

    /**
     * Add a multiple iid observation into this model
     * @param observationArray - the observation to be added - it has to be cast into proper datatype
     *                   depending on data we are working with
     */
    void addRange(Object[] observationArray)throws Exception;

    /**
     * Delete an existing observation from this component
     * @param observation - the observation to be removed
     */
    void remove(Object observation) throws Exception ;

    /**
     * Get the posterior distribution
     * @return return the current posterior distribution conditionally on the current set of observations
     */
    Object getPosterior() throws Exception;

    /**
     * Return the logarithm of the predictive density
     * @param observation - the value at which the density will be evaluated
     * @return the logarithm of the predictive probability evaluated at the point of input observation
     */
    double logPredictive(Object observation) throws Exception;

    /**
     * Compute the marginal likelihood, i.e., prob(data | hyperparameter) = int_{w} p(data,w | \lambda)
     * @return the logarithm of the marginal probability of the current set of observations
     */
    double logMarginal()throws Exception;

    /// <summary>
    /// Compute the expectation of  log likelihood, i.e., E(log(prob(data | hyperparameter)))
    /// </summary>
    /// <returns></returns>

    /**
     * Compute the expectation of  log likelihood, i.e., E(log(prob(data | hyperparameter)))
     * @param observation -the value at which the expectation will be evaluated
     * @return  the logarithm of the expectation of  log likelihood evaluated at the point of input observation
     */
    double expectationLogLikelihood(Object observation)throws Exception;

    /// <summary>
    /// Stochastic update in SVI
    /// </summary>
    /// <returns></returns>

    /**
     * Stochastic update in SVI
     * @param pp - the distribution to be updated into the current distribution
     * @param kappa - the learning rate
     */
    void stochasticUpdate(BayesianComponent pp, double kappa)throws  Exception;

    /**
     * Plus the given distribution to the current distribution
     * @param pp - the distribution to be added into the current distribution
     */
    void plus(BayesianComponent pp)throws  Exception;
    Object clone();
}
