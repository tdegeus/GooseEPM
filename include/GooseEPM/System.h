/**
 * @file
 * @copyright Copyright 2022. Marko Popovic, Tom de Geus. All rights reserved.
 * @license This project is released under the MIT License.
 */

#ifndef GOOSEEPM_SYSTEM_H
#define GOOSEEPM_SYSTEM_H

#include <prrng.h>
#include <xtensor/xset_operation.hpp>
#include <xtensor/xsort.hpp>

#include "config.h"
#include "version.h"

namespace GooseEPM {

namespace detail {

template <class T>
inline size_t argmax(const T& a)
{
    return std::distance(a.begin(), std::max_element(a.begin(), a.end()));
}

template <class T>
inline size_t argmin(const T& a)
{
    return std::distance(a.begin(), std::min_element(a.begin(), a.end()));
}

template <class T>
inline typename T::value_type mean(const T& a)
{
    double init = 0.0;
    return std::accumulate(a.begin(), a.end(), init) / static_cast<double>(a.size());
}

template <class T>
inline typename T::value_type amin(const T& a)
{
    return *min_element(a.begin(), a.end());
}

template <class T>
inline typename T::value_type amax(const T& a)
{
    return *max_element(a.begin(), a.end());
}

/**
 * @brief Create a distance lookup as follows:
 *
 *      [0, 1, 2, ..., N - 2, N - 1, -N + 1, -N + 2, ... -3, -2, -1]
 *
 * with `N` the shape of the propagator along the considered axis (`== distance.size()`).
 * For example for `N = 7` the input could be:
 *
 *      distance = [-3, -2, -1,  0,  1,  2,  3]
 *
 * The the lookup table (output) is:
 *
 *      // dist:    0   1   2   3   4   5   6  -6  -5  -4  -3  -2  -1
 *      lookup = [  3,  4,  5,  6,  0,  1,  2,  4,  5,  6,  0,  1,  2]
 *
 * whereby periodicity is applied such that:
 *
 *      // dist:    0   1   2   3   4   5   6  -6  -5  -4  -3  -2  -1
 *      // per:     0   1   2   3  -3  -2  -1   1   2   3  -3  -2  -1
 *
 * @param distance List with distances according to which an axis of the propagator is orders.
 * @return Look-up list.
 */
template <class T>
inline T create_distance_lookup(const T& distance)
{
    using value_type = typename T::value_type;
    static_assert(std::numeric_limits<value_type>::is_integer, "Distances must be integer");
    static_assert(std::numeric_limits<value_type>::is_signed, "Distances must be signed");
    GOOSEEPM_REQUIRE(distance.dimension() == 1, std::invalid_argument);

    value_type lower = detail::amin(distance);
    value_type upper = detail::amax(distance) + 1;
    auto d = xt::arange<value_type>(lower, upper);
    GOOSEEPM_REQUIRE(xt::all(xt::in1d(distance, d)), std::invalid_argument);
    GOOSEEPM_REQUIRE(distance.size() == upper - lower, std::invalid_argument);

    value_type N = static_cast<value_type>(distance.size());
    T ret = xt::empty<value_type>({2 * N - 1});

    for (value_type i = 0; i < upper; ++i) {
        ret(i) = detail::argmax(xt::equal(distance, i));
    }
    for (value_type i = upper; i < N; ++i) {
        ret(i) = detail::argmax(xt::equal(distance, i - N));
    }
    for (value_type i = -1; i >= lower; --i) {
        ret.periodic(i) = detail::argmax(xt::equal(distance, i));
    }
    for (value_type i = lower; i > -N; --i) {
        ret.periodic(i) = detail::argmax(xt::equal(distance, N + i));
    }

    return ret;
}

} // namespace detail

/**
 * @brief Athermal system that can be driving in imposed stress or imposed strain, as follows.
 *
 *      -   Imposed strain: take (an) event driven step(s) using
 *          eventDrivenStep() or eventDrivenSteps()
 *
 *      -   Imposed stress: take (a) failure step(s) using
 *          failureStep() or failureSteps()
 *
 * Note that by default the stress is taken homogenous.
 * To use a random stress field use (in Python code):
 *
 *      system = SystemAthermal(..., init_random_stress=True)
 *
 * which is simply shorthand for:
 *
 *      system = SystemAthermal()
 *      system.initSigmaPropogator(0.1)
 *
 * Note that this can be a bit costly: it find a compatible stress field by applying the convolution
 * between a random field and the propagator.
 * Instead, you can use a pre-computed field (that you can much faster compute by computing the
 * convolution in Fourier space) by:
 *
 *      system = SystemAthermal(..., init_random_stress=False)
 *      system.stress = ...
 *      system.sigbar = ...
 */
class SystemAthermal {

public:
    /**
     * @param propagator The propagator `[M, N]`.
     * @param distances_rows The distance that each row of the propagator corresponds to `[M]`.
     * @param distances_cols The distance that each column of the propagator corresponds to `[N]`.
     * @param sigmay_mean Mean yield stress for every block `[M, N]`.
     * @param sigmay_std Standard deviation of the yield stress for every block `[M, N]`.
     * @param seed Seed of the random number generator.
     * @param failure_rate Failure rate (irrelevant if event-driving protocol is used).
     * @param alpha Exponent characterising the shape of the potential.
     * @param sigmabar Mean stress to initialise the system.
     * @param fixed_stress If `true` the stress is kept constant.
     * @param init_random_stress If `true` a random compatible stress is initialised.
     */
    SystemAthermal(
        const array_type::tensor<double, 2>& propagator,
        const array_type::tensor<ptrdiff_t, 1>& distances_rows,
        const array_type::tensor<ptrdiff_t, 1>& distances_cols,
        const array_type::tensor<double, 2>& sigmay_mean,
        const array_type::tensor<double, 2>& sigmay_std,
        uint64_t seed,
        double failure_rate = 1,
        double alpha = 1.5,
        double sigmabar = 0,
        bool fixed_stress = false,
        bool init_random_stress = true)
    {
        GOOSEEPM_REQUIRE(propagator.dimension() == 2, std::out_of_range);
        GOOSEEPM_REQUIRE(distances_rows.size() == propagator.shape(0), std::out_of_range);
        GOOSEEPM_REQUIRE(distances_cols.size() == propagator.shape(1), std::out_of_range);
        GOOSEEPM_REQUIRE(xt::has_shape(sigmay_mean, propagator.shape()), std::out_of_range);
        GOOSEEPM_REQUIRE(xt::has_shape(sigmay_std, propagator.shape()), std::out_of_range);

        m_t = 0;
        m_failure_rate = failure_rate;
        m_alpha = alpha;
        m_fixed_stress = fixed_stress;
        m_propagator = propagator;
        m_drow = detail::create_distance_lookup(distances_rows);
        m_dcol = detail::create_distance_lookup(distances_cols);
        m_gen = prrng::pcg32(seed);
        m_epsp = xt::zeros<double>(propagator.shape());
        m_sigy = xt::empty<double>(propagator.shape());
        m_unstable = xt::zeros<bool>(propagator.shape());
        m_sigy_mu = sigmay_mean;
        m_sigy_std = sigmay_std;
        m_sigbar = sigmabar;

        for (size_t i = 0; i < m_sigy.size(); ++i) {
            m_sigy.flat(i) =
                m_gen.normal(std::array<size_t, 1>{1}, m_sigy_mu.flat(i), m_sigy_std.flat(i))(0);
        }

        if (init_random_stress) {
            m_sig = xt::empty<double>(propagator.shape());
            this->initSigmaPropogator(0.1);
            this->set_sigmabar(sigmabar);
        }
        else {
            m_sig = sigmabar * xt::ones<double>(propagator.shape());
        }

        m_unstable = xt::where(m_sig > 0, m_sig > m_sigy, m_sig < m_sigy);
    }

    /**
     * @brief Return the shape.
     * @return Shape
     */
    const auto& shape() const
    {
        return m_propagator.shape();
    }

    /**
     * @brief Set the time
     * @param t Time.
     */
    void set_t(double t)
    {
        m_t = t;
    }

    /**
     * @brief Get the time
     * @return Time.
     */
    uint64_t t() const
    {
        return m_t;
    }

    /**
     * @brief Set the state of the random number generator.
     * @param state State.
     */
    void set_state(uint64_t state)
    {
        m_gen.restore(state);
    }

    /**
     * @brief Get the state of the random number generator.
     * @return State.
     */
    uint64_t state() const
    {
        return m_gen.state();
    }

    /**
     * @brief Set the plastic strain.
     * @param epsp Plastic strain.
     */
    void set_epsp(const array_type::tensor<double, 2>& epsp)
    {
        m_epsp = epsp;
    }

    /**
     * @brief Get the plastic strain.
     * @return Plastic strain.
     */
    const array_type::tensor<double, 2>& epsp() const
    {
        return m_epsp;
    }

    /**
     * @brief Set the yield stress.
     * @note This function overrides the list of unstable blocks to currently unstable blocks.
     * @param sigmay Yield stress.
     */
    void set_sigmay(const array_type::tensor<double, 2>& sigmay)
    {
        m_sigy = sigmay;
        m_unstable = xt::where(m_sig > 0, m_sig > m_sigy, m_sig < m_sigy);
    }

    /**
     * @brief Get the yield stress.
     * @return Yield stress.
     */
    const array_type::tensor<double, 2>& sigmay() const
    {
        return m_sigy;
    }

    /**
     * @brief Set the stress.
     * If a fixed stress protocol is used, the fixed stress is set to `mean(sigma)`.
     * You can use SystemAthermal::set_sigmabar() to set the fixed stress manually.
     *
     * @note This function overrides the list of unstable blocks to currently unstable blocks.
     *
     * @param sigma Stress.
     */
    void set_sigma(const array_type::tensor<double, 2>& sigma)
    {
        m_sig = sigma;
        m_sigbar = detail::mean(m_sig);
        m_unstable = xt::where(m_sig > 0, m_sig > m_sigy, m_sig < m_sigy);
    }

    /**
     * @brief Get the stress.
     * @return Stress.
     */
    const array_type::tensor<double, 2>& sigma() const
    {
        return m_sig;
    }

    /**
     * @brief Set the imposed stress.
     *
     * @param sigmabar Imposed stress.
     */
    void set_sigmabar(double sigmabar)
    {
        m_sigbar = sigmabar;
        m_sig -= detail::mean(m_sig) - m_sigbar;
        m_unstable = m_unstable || (xt::where(m_sig > 0, m_sig > m_sigy, m_sig < m_sigy));
    }

    /**
     * @brief Get the average (imposed) stress.
     * @return Average stress.
     */
    double sigmabar() const
    {
        return m_sigbar;
    }

    /**
     * @brief Generate a stress field that is compatible, using a fast (approximative) technique.
     *
     * @param sigma_std Width of the normal distribution of stresses (mean == 0).
     * @param delta_r Distance to use, see above.
     */
    void initSigmaFast(double sigma_std, size_t delta_r)
    {
        m_sig.fill(0);
        ptrdiff_t d = static_cast<ptrdiff_t>(delta_r);

        for (ptrdiff_t i = 0; i < m_sig.shape(0); ++i) {
            for (ptrdiff_t j = 0; j < m_sig.shape(1); ++j) {
                double dsig = m_gen.normal(std::array<size_t, 1>{1}, 0, sigma_std)(0);
                m_sig(i, j) += dsig;
                m_sig.periodic(i - d, j) -= 0.5 * dsig;
                m_sig.periodic(i + d, j) -= 0.5 * dsig;
                m_sig.periodic(i, j - d) -= 0.5 * dsig;
                m_sig.periodic(i, j + d) -= 0.5 * dsig;
                m_sig.periodic(i + d, j + d) += 0.25 * dsig;
                m_sig.periodic(i - d, j + d) += 0.25 * dsig;
                m_sig.periodic(i + d, j - d) += 0.25 * dsig;
                m_sig.periodic(i - d, j - d) += 0.25 * dsig;
            }
        }
        m_sig *= 0.5;
    }

    /**
     * @brief Generate a stress field that is compatible. Internally the propagator is used.
     * Note that this computes the convolution between a random field and the propagator.
     * Since the propagator is defined in real space, this can ba quite costly.
     * Instead, it is advised you compute the convolution in Fourier space, using the expression
     * of the propagator in Fourier space.
     *
     * @param sigma_std Width of the normal distribution of stresses (mean == 0).
     */
    void initSigmaPropogator(double sigma_std)
    {
        m_sig.fill(0);

        for (ptrdiff_t i = 0; i < m_sig.shape(0); ++i) {
            for (ptrdiff_t j = 0; j < m_sig.shape(1); ++j) {

                double dsig = m_gen.normal(std::array<size_t, 1>{1}, 0, sigma_std)(0);

                for (ptrdiff_t k = 0; k < m_sig.shape(0); ++k) {
                    for (ptrdiff_t l = 0; l < m_sig.shape(1); ++l) {
                        if (i == k && j == l) {
                            m_sig(k, l) += dsig;
                        }
                        else {
                            m_sig(k, l) +=
                                m_propagator(m_drow.periodic(i - k), m_dcol.periodic(j - l)) * dsig;
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Make `n` makeFailureStep() calls.
     * @param n Number of steps to make.
     */
    void makeFailureSteps(size_t n)
    {
        for (size_t i = 0; i < n; ++i) {
            makeFailureStep();
        }
    }

    /**
     * @brief Fail an unstable block, chosen randomly.
     * @return Index of the failing particle (flat index).
     */
    size_t makeAthermalFailureStep()
    {
        auto failing = xt::argwhere(m_unstable);
        size_t nfailing = failing.size();

        m_t += m_gen.exponential(std::array<size_t, 1>{1}, m_failure_rate * nfailing)(0);

        size_t i = m_gen.randint(std::array<size_t, 1>{1}, static_cast<size_t>(nfailing - 1))(0);
        size_t idx = m_sig.shape(0) * failing[i][0] + failing[i][1];

        this->spatialParticleFailure(idx);
        return idx;
    }

    /**
     * @brief Fail weakest particle.
     * @return Index of the failing particle (flat index).
     */
    size_t makeWeakestFailureStep()
    {
        size_t idx = detail::argmin(xt::where(m_sig > 0, m_sigy - m_sig, m_sig - m_sigy));
        double x = m_sigy.flat(idx) - m_sig.flat(idx);

        if (m_sig.flat(idx) < 0) {
            x = -x;
        }

        if (x < 0) {
            m_t += 1.0;
        }
        else {
            m_t += std::exp(std::pow(200.0 * x, m_alpha)); // todo: why 200 x?
        }

        this->spatialParticleFailure(idx);
        return idx;
    }

    /**
     * @brief Fail a block.
     *
     * -    Change the stress in the block.
     * -    Stabilise the blocks.
     * -    Apply the propagator to change the stress in all 'surrounding' blocks.
     * -    Check if there are any new unstable blocks.
     *
     * @param idx Flat index of the block to fail.
     */
    void spatialParticleFailure(size_t idx)
    {
        double dsig = m_sig.flat(idx) + m_gen.normal(std::array<size_t, 1>{1}, 0.0, 0.01)(0);

        m_unstable.flat(idx) = false;
        m_sig.flat(idx) -= dsig;
        m_epsp.flat(idx) += dsig;
        m_sigy.flat(idx) =
            m_gen.normal(std::array<size_t, 1>{1}, m_sigy_mu.flat(idx), m_sigy_std.flat(idx))(0);

        auto index = xt::unravel_index(idx, m_sig.shape());
        ptrdiff_t i0 = static_cast<ptrdiff_t>(index[0]);
        ptrdiff_t j0 = static_cast<ptrdiff_t>(index[1]);

        for (ptrdiff_t i = 0; i < m_sig.shape(0); ++i) {
            for (ptrdiff_t j = 0; j < m_sig.shape(1); ++j) {
                if (i == i0 && j == j0) {
                    continue;
                }
                m_sig(i, j) +=
                    m_propagator(m_drow.periodic(i - i0), m_dcol.periodic(j - j0)) * dsig;
            }
        }

        if (!m_fixed_stress) {
            m_sigbar -= dsig / static_cast<double>(m_sig.size());
        }

        m_sig -= detail::mean(m_sig) - m_sigbar;
        m_unstable = m_unstable || (xt::where(m_sig > 0, m_sig > m_sigy, m_sig < m_sigy));
    }

    /**
     * @brief Take imposed shear step according the the event-driving protocol.
     */
    void shiftImposedShear()
    {
        double dsig = detail::amin(m_sigy - m_sig);
        m_sig += dsig;
        m_sigbar += dsig;
        m_unstable = m_unstable || (xt::where(m_sig > 0, m_sig > m_sigy, m_sig < m_sigy));
    }

    /**
     * @brief Take event driven step; shift the applied shear (by changing the stress) such that
     * the weakest particle fails, then relax the system until there are no more unstable blocks.
     */
    void eventDrivenStep()
    {
        this->shiftImposedShear();

        while (xt::any(m_unstable)) {
            this->makeWeakestFailureStep();
        }
    }

    /**
     * @brief Take `n` event driven steps.
     * @param n Number of steps to take.
     */
    void eventDrivenSteps(size_t n)
    {
        for (size_t i = 0; i < n; ++i) {
            this->eventDrivenStep();
        }
    }

protected:
    prrng::pcg32 m_gen; ///< Random number generator.
    array_type::tensor<double, 2> m_propagator; ///< Propagator.
    array_type::tensor<ptrdiff_t, 1> m_drow; ///< Lookup list: distance -> row in #m_propagator.
    array_type::tensor<ptrdiff_t, 1> m_dcol; ///< Lookup list: distance -> column in #m_propagator.
    array_type::tensor<double, 2> m_sig; ///< Stress.
    array_type::tensor<double, 2> m_sigy; ///< Yield stress.
    array_type::tensor<double, 2> m_sigy_mu; ///< Mean yield stress.
    array_type::tensor<double, 2> m_sigy_std; ///< Standard deviation of yield stress.
    array_type::tensor<double, 2> m_epsp; ///< Plastic strain.
    array_type::tensor<bool, 2> m_unstable; ///< `true` if the block is unstable.
    double m_t; ///< Time.
    double m_failure_rate; ///< Failure rate.
    double m_alpha; ///< Exponent characterising the shape of the potential.
    bool m_fixed_stress; ///< Flag indicating whether the stress is fixed.
    double m_sigbar; ///< Average stress.
    bool m_initstress; ///< Flag indicating whether the stress has to be initialised.
};

} // namespace GooseEPM

#endif
