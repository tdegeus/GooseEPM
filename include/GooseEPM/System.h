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
 * @brief Check that a distance vector is of the following form:
 *
 *      distance = [-3, -2, -1,  0,  1,  2,  3]
 *
 * @param distance List with distances according to which an axis of the propagator is orders.
 */
template <class T>
inline bool check_distances(const T& distance)
{
    using value_type = typename T::value_type;
    static_assert(std::numeric_limits<value_type>::is_integer, "Distances must be integer");
    static_assert(std::numeric_limits<value_type>::is_signed, "Distances must be signed");

    if (distance.dimension() != 1) {
        return false;
    }

    value_type lower = detail::amin(distance);
    value_type upper = detail::amax(distance) + 1;
    auto d = xt::arange<value_type>(lower, upper);

    if (!xt::all(xt::in1d(distance, d))) {
        return false;
    }

    if (distance.size() != upper - lower) {
        return false;
    }

    return true;
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
    GOOSEEPM_REQUIRE(detail::check_distances(distance), std::out_of_range);

    value_type lower = detail::amin(distance);
    value_type upper = detail::amax(distance) + 1;

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

protected:
    /**
     * @param propagator The propagator `[M, N]`.
     * @param distances_rows The distance that each row of the propagator corresponds to `[M]`.
     * @param distances_cols The distance that each column of the propagator corresponds to `[N]`.
     * @param sigmay_mean Mean yield stress for every block `[M, N]`.
     * @param sigmay_std Standard deviation of the yield stress for every block `[M, N]`.
     * @param seed Seed of the random number generator.
     * @param failure_rate Failure rate (irrelevant if event-driven protocol is used).
     * @param alpha Exponent characterising the shape of the potential.
     * @param sigmabar Mean stress to initialise the system.
     * @param fixed_stress If `true` the stress is kept constant.
     * @param init_random_stress If `true` a random compatible stress is initialised.
     * @param init_relax Relax the system initially.
     */
    void initSystemAthermal(
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
        bool init_random_stress = true,
        bool init_relax = true)
    {
        GOOSEEPM_REQUIRE(propagator.dimension() == 2, std::out_of_range);
        GOOSEEPM_REQUIRE(distances_rows.size() == propagator.shape(0), std::out_of_range);
        GOOSEEPM_REQUIRE(distances_cols.size() == propagator.shape(1), std::out_of_range);
        GOOSEEPM_REQUIRE(detail::check_distances(distances_rows), std::out_of_range);
        GOOSEEPM_REQUIRE(detail::check_distances(distances_cols), std::out_of_range);
        GOOSEEPM_REQUIRE(xt::has_shape(sigmay_mean, sigmay_std.shape()), std::out_of_range);

        auto i = xt::flatten_indices(xt::argwhere(xt::equal(distances_rows, 0)));
        auto j = xt::flatten_indices(xt::argwhere(xt::equal(distances_rows, 0)));
        GOOSEEPM_REQUIRE(i.size() == 1, std::out_of_range);
        GOOSEEPM_REQUIRE(j.size() == 1, std::out_of_range);
        m_propagator_origin = propagator(i(0), j(0));

        m_t = 0.0;
        m_failure_rate = failure_rate;
        m_alpha = alpha;
        m_fixed_stress = fixed_stress;
        m_gen = prrng::pcg32(seed);

        m_propagator = propagator;
        m_drow = distances_rows;
        m_dcol = distances_cols;

        auto shape = sigmay_mean.shape();
        m_epsp = xt::zeros<double>(shape);
        m_sigy = xt::empty<double>(shape);
        m_sigy_mu = sigmay_mean;
        m_sigy_std = sigmay_std;

        for (size_t i = 0; i < m_sigy.size(); ++i) {
            m_sigy.flat(i) =
                m_gen.normal(std::array<size_t, 1>{1}, m_sigy_mu.flat(i), m_sigy_std.flat(i))(0);
        }

        if (init_random_stress) {
            m_sig = xt::empty<double>(shape);
            this->initSigmaPropogator(0.1);
            this->set_sigmabar(sigmabar);
        }
        else {
            m_sig = sigmabar * xt::ones<double>(shape);
            m_sigbar = sigmabar;
        }

        if (init_relax) {
            this->relaxPreparation();
        }
    }

public:
    SystemAthermal() = default;

    /**
     * @copydoc SystemAthermal::initSystemAthermal
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
        bool init_random_stress = true,
        bool init_relax = true)
    {
        this->initSystemAthermal(
            propagator,
            distances_rows,
            distances_cols,
            sigmay_mean,
            sigmay_std,
            seed,
            failure_rate,
            alpha,
            sigmabar,
            fixed_stress,
            init_random_stress,
            init_relax);
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
    double t() const
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
     * @param sigmay Yield stress.
     */
    void set_sigmay(const array_type::tensor<double, 2>& sigmay)
    {
        m_sigy = sigmay;
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
     *
     * @note
     *      If a fixed stress protocol is used, the fixed stress is set to `mean(sigma)`.
     *      See SystemAthermal::set_sigmabar() to change the average (applied) stress.
     *
     * @param sigma Stress.
     */
    void set_sigma(const array_type::tensor<double, 2>& sigma)
    {
        m_sig = sigma;
        m_sigbar = detail::mean(m_sig);
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
     * @brief Adjust the average stress.
     *
     * @note
     *      If the fixed stress protocol is used,
     *      the average stress is fixed to the here specified value.
     *
     * @param sigmabar Imposed stress.
     */
    void set_sigmabar(double sigmabar)
    {
        m_sigbar = sigmabar;
        m_sig -= detail::mean(m_sig) - m_sigbar;
    }

    /**
     * @brief Get the average (imposed) stress.
     * @return Average stress.
     */
    double sigmabar() const
    {
        if (m_fixed_stress) {
            return m_sigbar;
        }
        else {
            return detail::mean(m_sig);
        }
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

                for (size_t k = 0; k < m_propagator.shape(0); ++k) {
                    for (size_t l = 0; l < m_propagator.shape(1); ++l) {
                        ptrdiff_t di = m_drow(k);
                        ptrdiff_t dj = m_dcol(l);
                        m_sig.periodic(i + di, j + dj) -= dsig * m_propagator(k, l);
                    }
                }
            }
        }

        m_sig /= xt::sqrt(xt::sum(xt::pow(m_propagator, 2.0)));
    }

    /**
     * @brief Make `n` makeAthermalFailureStep() calls.
     * @param n Number of steps to make.
     */
    void makeAthermalFailureSteps(size_t n)
    {
        for (size_t i = 0; i < n; ++i) {
            makeAthermalFailureStep();
        }
    }

    /**
     * @brief Fail an unstable block, chosen randomly.
     * @note If no block is unstable, nothing happens, and `-1` is returned.
     * @return Index of the failing particle (flat index).
     */
    ptrdiff_t makeAthermalFailureStep()
    {
        auto failing = xt::argwhere(m_sig < -m_sigy || m_sig > m_sigy);
        size_t nfailing = failing.size();


        if (nfailing == 0) {
            return -1;
        }

        m_t += m_gen.exponential(std::array<size_t, 1>{1}, 1.0 / (m_failure_rate * nfailing))(0);

        size_t i = 0;
        if (nfailing > 1) {
            size_t i = m_gen.randint(std::array<size_t, 1>{1}, static_cast<size_t>(nfailing - 1))(0);
        }

        size_t idx = m_sig.shape(0) * failing[i][0] + failing[i][1];

        this->spatialParticleFailure(idx);
        return static_cast<ptrdiff_t>(idx);
    }

    /**
     * @brief Fail weakest particle (also when it was not unstable).
     * @return Index of the failing particle (flat index).
     */
    size_t makeWeakestFailureStep()
    {
        size_t idx = detail::argmax(xt::abs(m_sig) - m_sigy);
        double x = std::abs(m_sig.flat(idx)) - m_sigy.flat(idx);

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

        m_epsp.flat(idx) -= dsig * m_propagator_origin;
        m_sigy.flat(idx) =
            m_gen.normal(std::array<size_t, 1>{1}, m_sigy_mu.flat(idx), m_sigy_std.flat(idx))(0);

        auto index = xt::unravel_index(idx, m_sig.shape());
        ptrdiff_t i0 = static_cast<ptrdiff_t>(index[0]);
        ptrdiff_t j0 = static_cast<ptrdiff_t>(index[1]);

        for (size_t i = 0; i < m_propagator.shape(0); ++i) {
            for (size_t j = 0; j < m_propagator.shape(1); ++j) {
                ptrdiff_t di = m_drow(i);
                ptrdiff_t dj = m_dcol(j);
                m_sig.periodic(i0 + di, j0 + dj) += m_propagator(i, j) * dsig;
            }
        }

        if (m_fixed_stress) {
            m_sig -= detail::mean(m_sig) - m_sigbar;
        }
    }

    /**
     * @brief Change the imposed shear such that one block fails in the direction of shear.
     * @warning If you call this from a system that is not relaxed, the system will unload.
     * @param direction Select positive (+1) or negative (-1) direction.
     */
    void shiftImposedShear(int direction = 1)
    {
        double dsig;

        if (direction > 0) {
            dsig = detail::amin(m_sigy - m_sig) + 2.0 * std::numeric_limits<double>::epsilon();
        }
        else {
            dsig = -detail::amin(m_sig + m_sigy) + 2.0 * std::numeric_limits<double>::epsilon();
        }

        m_sig += dsig;
    }

    /**
     * @brief Take event driven step; shift the applied shear (by changing the stress) such that
     * the weakest particle fails, then relax the system until there are no more unstable blocks.
     *
     * @param max_steps Maximum number of iterations to allow.
     * @param max_steps_is_error Throw `std::runtime_error` if `max_steps` is reached.
     * @return Number of iterations taken: `max_steps` corresponds to a failure to converge.
     */
    size_t eventDrivenStep(size_t max_steps = 1000000, bool max_steps_is_error = true)
    {
        this->shiftImposedShear();
        return this->relax(max_steps, max_steps_is_error);
    }

    /**
     * @brief Relax the system by calling makeWeakestFailureStep() until there are no more unstable
     * blocks.
     * @param max_steps Maximum number of iterations to allow.
     * @param max_steps_is_error Throw `std::runtime_error` if `max_steps` is reached.
     * @return Number of iterations taken: `max_steps` corresponds to a failure to converge.
     */
    size_t relax(size_t max_steps = 1000000, bool max_steps_is_error = true)
    {
        for (size_t i = 0; i < max_steps; ++i) {
            if (xt::all(m_sig >= -m_sigy && m_sig <= m_sigy)) {
                return i;
            }
            this->makeWeakestFailureStep();
        }

        if (max_steps_is_error) {
            throw std::runtime_error("Failed to converge.");
        }

        return max_steps;
    }

    /**
     * @brief Relax the system by failing one unstable block chosen at random.
     * @param max_steps Maximum number of iterations to allow.
     * @param max_steps_is_error Throw `std::runtime_error` if `max_steps` is reached.
     * @return Number of iterations taken: `max_steps` corresponds to a failure to converge.
     */
    size_t relaxPreparation(size_t max_steps = 1000000, bool max_steps_is_error = true)
    {
        double t = m_t;
        ptrdiff_t idx = 0;

        for (size_t i = 0; i < max_steps; ++i) {
            if (idx < 0) {
                m_t = t;
                return i;
            }
            idx = this->makeAthermalFailureStep();
        }

        if (max_steps_is_error) {
            throw std::runtime_error("Failed to converge.");
        }

        m_t = t;
        return max_steps;
    }

    /**
     * @brief Take `n` event driven steps.
     * @param n Number of steps to take.
     * @param max_steps Maximum number of iterations to allow.
     * @return Total number of iterations taken.
     */
    size_t eventDrivenSteps(size_t n, size_t max_steps = 1000000)
    {
        size_t iter = 0;

        for (size_t i = 0; i < n; ++i) {
            iter += this->eventDrivenStep(max_steps, true);
        }

        return iter;
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
    double m_t; ///< Time.
    double m_failure_rate; ///< Failure rate.
    double m_alpha; ///< Exponent characterising the shape of the potential.
    bool m_fixed_stress; ///< Flag indicating whether the stress is fixed.
    double m_sigbar; ///< Average stress.
    bool m_initstress; ///< Flag indicating whether the stress has to be initialised.
    double m_propagator_origin; ///< Value of the propagator at the origin.
};

/**
 * @brief Thermal system.
 */
class SystemThermal : public SystemAthermal {
public:
    SystemThermal() = default;

    /**
     * @copydoc SystemAthermal::initSystemAthermal
     * @param temperature Temperature.
     */
    SystemThermal(
        const array_type::tensor<double, 2>& propagator,
        const array_type::tensor<ptrdiff_t, 1>& distances_rows,
        const array_type::tensor<ptrdiff_t, 1>& distances_cols,
        const array_type::tensor<double, 2>& sigmay_mean,
        const array_type::tensor<double, 2>& sigmay_std,
        uint64_t seed,
        double temperature,
        double failure_rate = 1,
        double alpha = 1.5,
        double sigmabar = 0,
        bool fixed_stress = false,
        bool init_random_stress = true,
        bool init_relax = true)
    {
        this->initSystemAthermal(
            propagator,
            distances_rows,
            distances_cols,
            sigmay_mean,
            sigmay_std,
            seed,
            failure_rate,
            alpha,
            sigmabar,
            fixed_stress,
            init_random_stress,
            init_relax);

        m_temperature = temperature;
    }

    /**
     * @brief Get the applied temperature.
     */
    double temperature() const
    {
        return m_temperature;
    }

    /**
     * @brief Set the applied temperature.
     */
    void set_temperature(double temperature)
    {
        m_temperature = temperature;
    }

    /**
     * @brief Make a thermal failure step.
     *
     *      -   Unstable blocks are failed at a rate `failure_rate`.
     *      -   Stable blocks are failed a rate proportional to their energy barrier.
     */
    void makeThermalFailureStep()
    {
        double dt;
        double min_dt = std::numeric_limits<double>::max();
        double inv_failure_rate = 1.0 / m_failure_rate;
        double inv_temp = 1.0 / m_temperature;
        size_t idx;

        for (size_t i = 0; i < m_sig.size(); ++i) {

            if (std::abs(m_sig.flat(i)) > m_sigy.flat(i)) {
                dt = m_gen.exponential(std::array<size_t, 1>{1}, inv_failure_rate)(0);
            }
            else {
                dt = m_gen.exponential(
                    std::array<size_t, 1>{1},
                    std::exp(
                        std::pow(std::abs(m_sigy.flat(i) - m_sig.flat(i)), m_alpha) * inv_temp) *
                        inv_failure_rate)(0);
            }
            if (dt < min_dt) {
                min_dt = dt;
                idx = i;
            }
        }

        m_t += min_dt;
        this->spatialParticleFailure(idx);
    }

    /**
     * @brief Make `n` steps with SystemThermal::makeThermalFailureStep.
     * @param n Number of steps to take.
     */
    void makeThermalFailureSteps(size_t n)
    {
        for (size_t i = 0; i < n; ++i) {
            this->makeThermalFailureStep();
        }
    }

protected:
    double m_temperature; ///< Temperature.
};

} // namespace GooseEPM

#endif
