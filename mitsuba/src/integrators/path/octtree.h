#pragma once

#include <mitsuba/core/plugin.h>

#include <array>
#include <vector>

MTS_NAMESPACE_BEGIN

void atomicAdd(Spectrum *dest, const Spectrum &src) {
    for (int c = 0; c < SPECTRUM_SAMPLES; ++c)
        atomicAdd(&(*dest)[c], src[c]);
}

class Octtree {
public:
    static constexpr int HISTOGRAM_RESOLUTION = 4;
    static constexpr int BIN_COUNT = HISTOGRAM_RESOLUTION * HISTOGRAM_RESOLUTION;

    struct Configuration {
        Float minimumLeafWeightForSampling = 40000;
        Float minimumLeafWeightForTraining = 20000;
        Float leafDecay = 0; /// set to 0 for hard reset after an iteration, 1 for no reset at all
        long maxNodeCount = 0;
    };
    
    struct TrainingNode {
        void decay(Float decayFactor) {
            m_lrWeight *= decayFactor;
            m_lrFirstMoment *= decayFactor;
            m_lrSecondMoment *= decayFactor;
            m_lrCost *= decayFactor;
        }

        TrainingNode &operator+=(const TrainingNode &other) {
            m_lrWeight += other.m_lrWeight;
            m_lrFirstMoment += other.m_lrFirstMoment;
            m_lrSecondMoment += other.m_lrSecondMoment;
            m_lrCost += other.m_lrCost;
            return *this;
        }

        Float getWeight() const {
            return m_lrWeight;
        }

        Spectrum getLrEstimate() const {
            return m_lrWeight > 0 ? m_lrFirstMoment / m_lrWeight : Spectrum(0.f);
        }

        Spectrum getLrSecondMoment() const {
            return m_lrWeight > 0 ? m_lrSecondMoment / m_lrWeight : Spectrum(0.f);
        }

        Spectrum getLrVariance() const {
            if (m_lrWeight == 0)
                return Spectrum(0.f);
            
            Spectrum result;
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i) {
                result[i] = std::max(m_lrSecondMoment[i] / m_lrWeight - Float(std::pow(m_lrFirstMoment[i] / m_lrWeight, 2)), Float(0));
            }
            return result;
        }

        Float getLrCost() const {
            return m_lrWeight > 0 ? m_lrCost / m_lrWeight : 0;
        }

        void splatLrEstimate(const Spectrum &sum, const Spectrum &sumSquares, Float cost, Float weight) {
            atomicAdd(&m_lrFirstMoment, sum);
            atomicAdd(&m_lrSecondMoment, sumSquares);
            atomicAdd(&m_lrCost, cost);
            atomicAdd(&m_lrWeight, weight);
        }

    private:
        Float m_lrWeight { 0.f };
        Spectrum m_lrFirstMoment { 0.f };
        Spectrum m_lrSecondMoment { 0.f };
        Float m_lrCost { 0.f };
    };
    
    struct SamplingNode {
        bool isValid() const { return m_isValid; }
        void learnFrom(const TrainingNode &trainingNode, const Configuration &config) {
            m_isValid = trainingNode.getWeight() >= config.minimumLeafWeightForSampling;
            
            if (trainingNode.getWeight() > 0) {
                lrEstimate = trainingNode.getLrEstimate();

                if (trainingNode.getLrCost() > 0) {
                    earsFactorR = trainingNode.getLrSecondMoment() / trainingNode.getLrCost();
                    earsFactorS = trainingNode.getLrVariance() / trainingNode.getLrCost();
                } else {
                    /// there can be caches where no work is done
                    /// (e.g., failed strict normals checks meaning no NEE samples or BSDF samples are ever taken)
                    earsFactorR = Spectrum(0.f);
                    earsFactorS = Spectrum(0.f);
                }
            }
        }
        
        Spectrum lrEstimate;
        Spectrum earsFactorR; // sqrt(2nd-moment / cost)
        Spectrum earsFactorS; // sqrt(variance / cost)
    
    private:
        bool m_isValid;
    };
    
    Configuration configuration;

    void setMaximumMemory(long bytes) {
        configuration.maxNodeCount = bytes / sizeof(Node);
    }
    
private:
    typedef uint32_t NodeIndex;
    
    struct Node {
        struct Child {
            NodeIndex index { 0 };
            std::array<TrainingNode, BIN_COUNT> training;
            std::array<SamplingNode, BIN_COUNT> sampling;
            
            bool isLeaf() const { return index == 0; }
            Float maxTrainingWeight() const {
                Float weight = 0;
                for (const auto &t : training)
                    weight = std::max(weight, t.getWeight());
                return weight;
            }
        };
        
        std::array<Child, 8> children;
    };
    
    std::vector<Node> m_nodes;
    
    int stratumIndex(Vector3f &pos) {
        int index = 0;
        for (int dim = 0; dim < 3; ++dim) {
            int bit = pos[dim] >= 0.5f;
            index |= bit << dim;
            pos[dim] = pos[dim] * 2 - bit;
        }
        return index;
    }
    
    NodeIndex splitNodeIfNecessary(Float weight) {
        if (weight < configuration.minimumLeafWeightForTraining)
            /// splitting not necessary
            return 0;
        
        if (configuration.maxNodeCount && long(m_nodes.size()) > configuration.maxNodeCount)
            /// we have already reached the maximum node number
            return 0;
        
        NodeIndex newNodeIndex = NodeIndex(m_nodes.size());
        m_nodes.emplace_back();
        
        for (int stratum = 0; stratum < 8; ++stratum) {
            /// split recursively if needed
            NodeIndex newChildIndex = splitNodeIfNecessary(weight / 8);
            m_nodes[newNodeIndex].children[stratum].index = newChildIndex;
        }
        
        return newNodeIndex;
    }
    
    std::array<TrainingNode, BIN_COUNT> build(NodeIndex index, bool needsSplitting) {
        std::array<TrainingNode, BIN_COUNT> sum;
        
        for (int stratum = 0; stratum < 8; ++stratum) {
            if (m_nodes[index].children[stratum].isLeaf()) {
                if (needsSplitting) {
                    NodeIndex newChildIndex = splitNodeIfNecessary(
                        m_nodes[index].children[stratum].maxTrainingWeight()
                    );
                    m_nodes[index].children[stratum].index = newChildIndex;
                }
            } else {
                /// build recursively
                auto buildResult = build(
                    m_nodes[index].children[stratum].index,
                    needsSplitting
                );
                m_nodes[index].children[stratum].training = buildResult;
            }
            
            auto &child = m_nodes[index].children[stratum];
            for (int bin = 0; bin < BIN_COUNT; ++bin) {
                sum[bin] += child.training[bin];
                child.sampling[bin].learnFrom(child.training[bin], configuration);
                child.training[bin].decay(configuration.leafDecay);
            }
        }
        
        return sum;
    }
    
public:
    Octtree() {
        m_nodes.emplace_back();

        /// initialize tree to some depth
        for (int stratum = 0; stratum < 8; ++stratum) {
            NodeIndex newChildIndex = splitNodeIfNecessary(
                8 * configuration.minimumLeafWeightForSampling
            );
            m_nodes[0].children[stratum].index = newChildIndex;
        }
    }
    
    /**
     * Accumulates all the data from training into the sampling nodes, refines the tree and resets the training nodes.
     */
    void build(bool needsSplitting) {
        auto sum = build(0, needsSplitting);
        m_nodes.shrink_to_fit();

        Float weightSum = 0;
        for (int bin = 0; bin < BIN_COUNT; ++bin)
            weightSum += sum[bin].getWeight();

        SLog(EInfo, "Octtree built [%ld samples, %ld nodes, %.1f MiB]",
            long(weightSum),
            m_nodes.size(),
            m_nodes.capacity() * sizeof(Node) / (1024.f * 1024.f)
        );
    }
    
    void lookup(Vector3f pos, int bin, const SamplingNode* &sampling, TrainingNode* &training) {
        NodeIndex currentNodeIndex = 0;
        while (true) {
            int stratum = stratumIndex(pos);
            auto &child = m_nodes[currentNodeIndex].children[stratum];
            if (currentNodeIndex == 0 || child.sampling[bin].isValid())
                /// a valid node for sampling
                sampling = &child.sampling[bin];
            
            if (child.isLeaf()) {
                /// reached a leaf node
                training = &child.training[bin];
                break;
            }
            
            currentNodeIndex = child.index;
        }
    }
};

MTS_NAMESPACE_END
