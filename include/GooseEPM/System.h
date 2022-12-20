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

/**
 * @brief Reogranize a propagator such that the distances are ordered.
 *
 * @param propagator Propagator to be reorganized.
 * @param drow Distance of each row.
 * @param dcol Distance of each column.
 * @return Reorganized propagator.
 */
template <class T, class D>
inline T reorganise_popagator(const T& propagator, const D& drow, const D& dcol)
{
    T ret = xt::empty_like(propagator);
    auto dr = xt::argsort(drow);
    auto dc = xt::argsort(dcol);

    for (size_t i = 0; i < propagator.shape(0); ++i) {
        for (size_t j = 0; j < propagator.shape(1); ++j) {
            ret(i, j) = propagator(dr(i), dc(j));
        }
    }

    return ret;
}

} // namespace detail

/**
 * @brief Athermal system that can be driven in imposed stress or imposed strain.
 *
 * **Imposed stress**:
 *
 *  -   The propagator \f$ G_{ij} \f$ must follow \f$ \sum\limits_{ij} G_{ij} = 0 \f$ and
 *      \f$ G(\Delta i = 0, \Delta j = 0) = -1 \f$.
 *      See SystemAthermal::propogator_follows_conventions.
 *
 * @note The average stress is not fixed internally. Consequently, numerical errors may accumulate.
 * Use SystemAthermal::set_sigmabar to correct if needed.
 *
 * **Imposed strain**:
 *
 *  -   The propagator \f$ G_{ij} \f$ must follow \f$ \sum\limits_{ij} G_{ij} = - 1 / N \f$,
 *      with \f$ N \f$ the size of the propagator, and \f$ G(\Delta i = 0, \Delta j = 0) = -1 \f$.
 *      See SystemAthermal::propogator_follows_conventions.
 *
 *  -   Driving proceeds by imposing the strain, that changes the stress through elasticity.
 *      To change the strain such that that the system just reaches the next yielding event,
 *      use SystemAthermal::shiftImposedShear.
 *
 * **Initialization**:
 *
 * By default the stress is chosen randomly according to compatibility, and the system is relaxed to
 * being stable. For customisation you can (in Python code):
 *
 *      system = SystemAthermal(..., init_random_stress=False, init_relax=False)
 *      system.sigma = ...
 *      ...
 *
 * Since initialisation can be a bit expensive, you can use the above to re-use the same initial
 * stress distribution in multiple simulations.
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
     * @param failure_rate Failure rate \f$ f_0 \f$.
     * @param alpha Exponent characterising the shape of the potential.
     * @param sigmabar Mean stress to initialise the system.
     * @param init_random_stress If `true` a random compatible stress is initialised.
     * @param init_relax Relax the system initially.
     *
     * @todo The failure rate might be rather a time. This should be clarified and documented.
     */
    void initSystemAthermal(
        const array_type::tensor<double, 2>& propagator,
        const array_type::tensor<ptrdiff_t, 1>& distances_rows,
        const array_type::tensor<ptrdiff_t, 1>& distances_cols,
        const array_type::tensor<double, 2>& sigmay_mean,
        const array_type::tensor<double, 2>& sigmay_std,
        uint64_t seed,
        double failure_rate,
        double alpha,
        double sigmabar,
        bool init_random_stress,
        bool init_relax)
    {
        GOOSEEPM_REQUIRE(propagator.dimension() == 2, std::out_of_range);
        GOOSEEPM_REQUIRE(distances_rows.size() == propagator.shape(0), std::out_of_range);
        GOOSEEPM_REQUIRE(distances_cols.size() == propagator.shape(1), std::out_of_range);
        GOOSEEPM_REQUIRE(detail::check_distances(distances_rows), std::out_of_range);
        GOOSEEPM_REQUIRE(detail::check_distances(distances_cols), std::out_of_range);
        GOOSEEPM_REQUIRE(xt::has_shape(sigmay_mean, sigmay_std.shape()), std::out_of_range);

        auto i = xt::argwhere(xt::equal(distances_rows, 0));
        auto j = xt::argwhere(xt::equal(distances_rows, 0));
        GOOSEEPM_REQUIRE(i.size() == 1, std::out_of_range);
        GOOSEEPM_REQUIRE(j.size() == 1, std::out_of_range);
        m_propagator_origin = propagator(i[0][0], j[0][0]);

        m_failure_rate = failure_rate;
        m_alpha = alpha;
        m_gen.seed(seed);

        m_propagator = detail::reorganise_popagator(propagator, distances_rows, distances_cols);
        m_drow = detail::amin(distances_rows);
        m_dcol = detail::amin(distances_cols);

        m_sigy_mu = sigmay_mean;
        m_sigy_std = sigmay_std;
        m_sig = xt::ones_like(m_sigy_mu);
        m_epsp = xt::zeros_like(m_sigy_mu);
        m_sigy = xt::empty_like(m_sigy_mu);
        m_sig *= sigmabar;

        for (size_t i = 0; i < m_sigy.size(); ++i) {
            m_sigy.flat(i) = m_gen.normal(m_sigy_mu.flat(i), m_sigy_std.flat(i));
        }

        if (init_random_stress) {
            this->initSigmaPropogator(0.1);
            this->set_sigmabar(sigmabar);
        }

        if (init_relax) {
            this->relaxAthermal();
        }

        m_t = 0.0;
        m_epsp.fill(0.0);
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
            init_random_stress,
            init_relax);
    }

    /**
     * @brief Check if the propagator is consistent with the conventions of this class
     * (see discussion in SystemAthermal).
     *
     * @param protocol Select imposed `"stress"` or `"strain"`.
     */
    bool propogator_follows_conventions(const std::string& protocol) const
    {
        if (!xt::allclose(m_propagator_origin, -1)) {
            return false;
        }

        if (protocol == "stress") {
            return xt::allclose(xt::mean(m_propagator), 0.0);
        }
        else if (protocol == "strain") {
            return xt::allclose(
                xt::mean(m_propagator), -1.0 / static_cast<double>(m_propagator.size()));
        }

        throw std::out_of_range("Unknown protocol");
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
     * @brief Set the propagator.
     * @param propagator Propagator.
     */
    void set_propagator(const array_type::tensor<double, 2>& propagator)
    {
        m_propagator = propagator;
    }

    /**
     * @brief Get the propagator.
     * @return Propagator.
     */
    const array_type::tensor<double, 2>& propagator() const
    {
        return m_propagator;
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
     * @brief Set the mean yield stress of each block.
     * @param sigmay Mean yield stress.
     */
    void set_sigmay_mean(const array_type::tensor<double, 2>& sigmay_mean)
    {
        m_sigy_mu = sigmay_mean;
    }

    /**
     * @brief Get the mean yield stress of each block.
     * @return Mean yield stress.
     */
    const array_type::tensor<double, 2>& sigmay_mean() const
    {
        return m_sigy_mu;
    }

    /**
     * @brief Set the yield stress standard deviation of each block.
     * @param sigmay_std Yield stress standard deviation.
     */
    void set_sigmay_std(const array_type::tensor<double, 2>& sigmay_std)
    {
        m_sigy_std = sigmay_std;
    }

    /**
     * @brief Get the yield stress standard deviation of each block.
     * @return Yield stress standard deviation.
     */
    const array_type::tensor<double, 2>& sigmay_std() const
    {
        return m_sigy_std;
    }

    /**
     * @brief Set the stress.
     * @param sigma Stress.
     */
    void set_sigma(const array_type::tensor<double, 2>& sigma)
    {
        m_sig = sigma;
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
     *      If the propagator is defined according to the fixed stress protocol,
     *      the average stress is fixed to the here specified value.
     *
     * @param sigmabar Imposed stress.
     */
    void set_sigmabar(double sigmabar)
    {
        m_sig -= detail::mean(m_sig) - sigmabar;
    }

    /**
     * @brief Get the average stress.
     * @return Average stress.
     */
    double sigmabar() const
    {
        return detail::mean(m_sig);
    }

    /**
     * @brief Generate a stress field that is compatible, using a fast but approximative technique.
     *
     * @param sigma_std Width of the normal distribution of stresses (mean == 0).
     * @param delta_r Distance to use, see above.
     */
    void initSigmaFast(double sigma_std, size_t delta_r)
    {
        m_sig.fill(0);
        ptrdiff_t d = static_cast<ptrdiff_t>(delta_r);
        auto dsig = m_gen.normal<decltype(m_sig)>(m_sig.shape(), 0, sigma_std);

        for (ptrdiff_t i = 0; i < m_sig.shape(0); ++i) {
            for (ptrdiff_t j = 0; j < m_sig.shape(1); ++j) {
                m_sig(i, j) += dsig(i, j);
                m_sig.periodic(i - d, j) -= 0.5 * dsig(i, j);
                m_sig.periodic(i + d, j) -= 0.5 * dsig(i, j);
                m_sig.periodic(i, j - d) -= 0.5 * dsig(i, j);
                m_sig.periodic(i, j + d) -= 0.5 * dsig(i, j);
                m_sig.periodic(i + d, j + d) += 0.25 * dsig(i, j);
                m_sig.periodic(i - d, j + d) += 0.25 * dsig(i, j);
                m_sig.periodic(i + d, j - d) += 0.25 * dsig(i, j);
                m_sig.periodic(i - d, j - d) += 0.25 * dsig(i, j);
            }
        }
        m_sig *= 0.5;
    }

    /**
     * @brief Generate a stress field that is compatible. Internally the propagator is used.
     *
     * @note
     *      This protocol computes the convolution between a random field and the propagator.
     *      Since the propagator is defined in real space, this can ba quite costly.
     *      Instead, it is advised that you compute the convolution in Fourier space,
     *      using the expression of the propagator in Fourier space.
     *      Any additions to the Python module to this extent are welcome!
     *
     * @param sigma_std Width of the normal distribution of stresses (mean == 0).
     */
    void initSigmaPropogator(double sigma_std)
    {
        m_sig.fill(0);
        auto dsig = m_gen.normal<decltype(m_sig)>(m_sig.shape(), 0, sigma_std);

        for (ptrdiff_t i = 0; i < m_sig.shape(0); ++i) {
            for (ptrdiff_t j = 0; j < m_sig.shape(1); ++j) {
                for (ptrdiff_t k = 0; k < m_propagator.shape(0); ++k) {
                    for (ptrdiff_t l = 0; l < m_propagator.shape(1); ++l) {
                        m_sig.periodic(i + m_drow + k, j + m_dcol + l) -=
                            dsig(i, j) * m_propagator(k, l);
                    }
                }
            }
        }

        m_sig /= xt::sqrt(xt::sum(xt::pow(m_propagator, 2.0)));
    }

    /**
     * @brief Fail an unstable block, chosen randomly from all unstable blocks.
     * The time is advanced by \f$ \exp( 1 / (f_0 n) )\f$ with \f$ f_0 \f$ the failure rate
     * (see constructor) and \f$ n \f$ the number of unstable blocks.
     *
     * @note If no block is unstable, nothing happens, and `-1` is returned.
     *
     * @return Index of the failing particle (flat index).
     */
    ptrdiff_t makeAthermalFailureStep()
    {
        auto failing = xt::argwhere(xt::abs(m_sig) >= m_sigy);
        size_t nfailing = failing.size();

        if (nfailing == 0) {
            return -1;
        }

        m_t += m_gen.exponential() / (m_failure_rate * nfailing);

        size_t i = 0;
        if (nfailing > 1) {
            i = m_gen.randint(nfailing);
        }

        size_t idx = m_sig.shape(0) * failing[i][0] + failing[i][1];

        this->spatialParticleFailure(idx);
        return static_cast<ptrdiff_t>(idx);
    }

    /**
     * @brief Fail weakest block unstable and advance the time by one.
     *
     * @note If no block is unstable, nothing happens, and `-1` is returned.
     *
     * @return Index of the failing particle (flat index).
     */
    ptrdiff_t makeWeakestFailureStep()
    {
        size_t idx = detail::argmax(xt::abs(m_sig) - m_sigy);
        double x = m_sigy.flat(idx) - std::abs(m_sig.flat(idx));

        if (x > 0) {
            return -1;
        }

        m_t += 1.0;
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
        double dsig = m_sig.flat(idx) + m_gen.normal(0.0, 0.01);

        m_epsp.flat(idx) -= dsig * m_propagator_origin;
        m_sigy.flat(idx) = m_gen.normal(m_sigy_mu.flat(idx), m_sigy_std.flat(idx));

        auto index = xt::unravel_index(idx, m_sig.shape());
        ptrdiff_t di = static_cast<ptrdiff_t>(index[0]) + m_drow;
        ptrdiff_t dj = static_cast<ptrdiff_t>(index[1]) + m_dcol;

        for (ptrdiff_t i = 0; i < m_propagator.shape(0); ++i) {
            for (ptrdiff_t j = 0; j < m_propagator.shape(1); ++j) {
                m_sig.periodic(di + i, dj + j) += m_propagator(i, j) * dsig;
            }
        }
    }

    /**
     * @brief Change the imposed shear such that one block fails in the direction of shear.
     *
     * @warning
     *      If you call this from a system that is not relaxed, the system will move in the
     *      opposite direction than imposed.
     *
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
     * @brief Relax the system by calling SystemAthermal::makeAthermalFailureStep
     * until there are no more unstable blocks.
     *
     * @param max_steps Maximum number of iterations to allow.
     * @param max_steps_is_error If `true`, throw `std::runtime_error` if `max_steps` is reached.
     * @return Number of iterations taken: `max_steps` corresponds to a failure to converge.
     */
    size_t relaxAthermal(size_t max_steps = 1000000, bool max_steps_is_error = true)
    {

        for (size_t i = 0; i < max_steps; ++i) {
            auto idx = this->makeAthermalFailureStep();
            if (idx < 0) {
                return i;
            }
        }

        if (max_steps_is_error) {
            throw std::runtime_error("Failed to converge.");
        }

        return max_steps;
    }

    /**
     * @brief Relax the system by calling SystemAthermal::makeWeakestFailureStep
     * until there are no more unstable blocks.
     *
     * @param max_steps Maximum number of iterations to allow.
     * @param max_steps_is_error If `true`, throw `std::runtime_error` if `max_steps` is reached.
     * @return Number of iterations taken: `max_steps` corresponds to a failure to converge.
     */
    size_t relaxWeakest(size_t max_steps = 1000000, bool max_steps_is_error = true)
    {

        for (size_t i = 0; i < max_steps; ++i) {
            auto idx = this->makeWeakestFailureStep();
            if (idx < 0) {
                return i;
            }
        }

        if (max_steps_is_error) {
            throw std::runtime_error("Failed to converge.");
        }

        return max_steps;
    }

protected:
    prrng::pcg32 m_gen; ///< Random number generator.
    array_type::tensor<double, 2> m_propagator; ///< Propagator.
    ptrdiff_t m_drow; ///< Minimal distance in row direction, used as offset.
    ptrdiff_t m_dcol; ///< Minimal distance in column direction, used as offset.
    array_type::tensor<double, 2> m_sig; ///< Stress.
    array_type::tensor<double, 2> m_sigy; ///< Yield stress.
    array_type::tensor<double, 2> m_sigy_mu; ///< Mean yield stress.
    array_type::tensor<double, 2> m_sigy_std; ///< Standard deviation of yield stress.
    array_type::tensor<double, 2> m_epsp; ///< Plastic strain.
    double m_t; ///< Time.
    double m_failure_rate; ///< Failure rate.
    double m_alpha; ///< Exponent characterising the shape of the potential.
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
     * Time is advanced by the smallest time to the next failure. Thereby:
     *
     *  -   Unstable blocks are failed at a rate `failure_rate`.
     *  -   Stable blocks are failed a rate proportional to their energy barrier.
     */
    void makeThermalFailureStep()
    {
        double inv_failure_rate = 1.0 / m_failure_rate;
        double inv_temp = 1.0 / m_temperature;
        size_t idx;

        auto dt = m_gen.exponential<decltype(m_sig)>(m_sig.shape());
        auto scale = xt::where(
            xt::abs(m_sig) >= m_sigy,
            inv_failure_rate,
            xt::exp(xt::pow(m_sigy - xt::abs(m_sig), m_alpha) * inv_temp) * inv_failure_rate);

        dt *= scale;
        idx = detail::argmin(dt);
        m_t += dt.flat(idx);
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
