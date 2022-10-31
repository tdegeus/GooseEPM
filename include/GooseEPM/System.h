/**
 * @file version.h
 * @copyright Copyright 2022. Tom de Geus. All rights reserved.
 * @license This project is released under the MIT License.
 */

#ifndef GOOSEEPM_SYSTEM_H
#define GOOSEEPM_SYSTEM_H

#include <prrng.h>
#include <vector>

#include "config.h"
#include "version.h"

namespace GooseEPM {

class SystemAthermal {

public:
    template <class T, class S, class Y, class Z>
    SystemAthermal(
        const T& propagator,
        const Y& sigmay_mean,
        const Z& sigmay_std,
        const S& sigmay_initstate,
        uint64_t seed)
    {
        GOOSEEPM_REQUIRE(propagator.ndim() == 2, std::out_of_range);
        GOOSEEPM_REQUIRE(xt::has_shape(sigmay_initstate, propagator.shape()), std::out_of_range);
        GOOSEEPM_REQUIRE(xt::has_shape(sigmay_mean, propagator.shape()), std::out_of_range);
        GOOSEEPM_REQUIRE(xt::has_shape(sigmay_std, propagator.shape()), std::out_of_range);

        m_t = 0;
        m_propagator = propagator;
        m_sigy_gen = prrng::pcg32_tensor<2>(sigmay_initstate);
        m_gen = prrng::pcg32(seed);
        m_sig = xt::zeros<double>(sigmay_initstate.shape());
        m_epsp = xt::zeros<double>(sigmay_initstate.shape());
        m_sigy = xt::empty<double>(sigmay_initstate.shape());
        m_sigy_mu = sigmay_mean;
        m_sigy_std = sigmay_std;

        for (size_t i = 0; i < m_sigy.size(); ++i) {
            m_sigy.flat(i) = m_sigy_gen.flat(i).normal({}, m_sigy_mu.flat(i), m_sigy_std.flat(i));
        }

        if (m_propagator.shape(0) % 2 == 0) {
            m_imid = m_propagator.shape(0) / 2;
        }
        else {
            m_imid = (m_propagator.shape(0) - 1) / 2;
        }

        if (m_propagator.shape(1) % 2 == 0) {
            m_jmid = m_propagator.shape(1) / 2;
        }
        else {
            m_jmid = (m_propagator.shape(1) - 1) / 2;
        }
    }

    /**
     * @brief Restore random number generators.
     * @param sigmay_state State of the random number generator for the sigmay.
     * @param state State of the general purpose random number generator.
     */
    template <class T>
    void restore(const T& sigmay_state, uint64_t state)
    {
        GOOSEEPM_REQUIRE(xt::has_shape(sigmay_state, m_sigy_gen.shape(), std::out_of_range));
        m_sigy_gen.restore(sigmay_state);
        m_gen.restore(state);
    }

    size_t makeAthermalFailureStep()
    {
        auto failing = xt::argwhere(m_sig < -m_sigy || m_sig > m_sigy);
        size_t nfailing = failing.size();
        m_t += m_gen.expontial({}, m_failure_rate * nfailing);
        auto i = randomNumberGenerator.randint({}, nfailing - 1);
        size_t idx = m_shape(0) * failing[i][0] + failing[i][1];
        this->spatialParticleFailure(idx);
        return idx;
    }

    size_t makeWeakestFailureStep()
    {
        size_t idx = xt::argmin(m_sigy - m_sig);
        double x = m_sigy.flat(idx) - m_sig.flat(idx);

        if (x < 0) {
            m_t += 1.0;
        }
        else {
            m_t += std::exp(std::pow(200.0 * x, m_alpha));
        }

        this->spatialParticleFailure(idx);
        return idx;
    }

    /**
     * @brief Fail a block.
     *
     * -    Change the stress in the block and stabilise it.
     * -    Apply the propagator to change the stress in all 'surrounding' blocks.
     *
     * @param idx Flat index of the block to fail.
     */
    void spatialParticleFailure(size_t idx)
    {
        double dsig = m_sig.flat(idx) + m_gen.normal({}, 0.0, 0.01);

        m_sig.flat(idx) -= dsig;
        m_epsp.flat(idx) += dsig;
        m_sigy.flat(idx) = m_sigy_gen.flat(i).normal({}, m_sigy_mu.flat(i), m_sigy_std.flat(i));

        auto index = xt::unravel_index(idx, m_sig.shape());
        size_t i0 = index[0];
        size_t j0 = index[1];

        for (size_t i = 0; i < m_sig.shape(0); ++i) {
            for (size_t j = 0; j < m_sig.shape(1); ++j) {
                if (i == i0 && j == j0) {
                    continue;
                }
                size_t di = this->propagator_index(i0, i, m_imid);
                size_t dj = this->propagator_index(j0, j, m_jmid);
                m_sig(i, j) += m_propagator(di, dj) * dsig;
            }
        }

        if (!m_fixed_stress) {
            m_sigbar -= dsig / static_cast<double>(m_sig.size());
        }

        m_sig -= xt::mean(m_sig)() - m_sigbar;
    }

    double nextYieldStress(size_t idx)
    {
        double r = m_sigy_gen.flat(idx).next_double();
        m_normal[i][j].quantile(r);
        return r;
    }

private:
    size_t propagator_index(size_t i0, size_t i, size_t imid)
    {
        if (i < i0) {
            size_t d = i0 - i;
            if (d > imid) {
                return imid + (imid - d);
            }
            else {
                return imid - d;
            }
        }
        else {
            size_t d = i - i0;
            if (d > imid) {
                return imid + (imid - d);
            }
            else {
                return imid + d;
            }
        }
    }

protected:
    prrng::pcg32 m_gen;
    prrng::pcg32_tensor<2> m_sigy_gen;
    array_type::tensor<double, 2> m_propagator;
    array_type::tensor<double, 2> m_sig;
    array_type::tensor<double, 2> m_sigy;
    array_type::tensor<double, 2> m_sigy_mu;
    array_type::tensor<double, 2> m_sigy_std;
    array_type::tensor<double, 2> m_epsp;
    double m_t;
    size_t m_imid;
    size_t m_jmid;
}

} // namespace GooseEPM

#endif
