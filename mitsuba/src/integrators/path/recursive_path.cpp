/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <OpenImageDenoise/oidn.hpp>

#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/statistics.h>

#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <mutex>

#include "octtree.h"

/// we support outputting several AOVs that can be helpful for research and debugging.
/// since they are computationally expensive, we disable them by default.
/// uncomment the following line to enable outputting AOVs:
//#define EARS_INCLUDE_AOVS

#include "recursive_path_aovs.h"

MTS_NAMESPACE_BEGIN

thread_local StatsRecursiveImageBlockCache *StatsRecursiveImageBlockCache::instance = nullptr;
thread_local StatsRecursiveDescriptorCache *StatsRecursiveDescriptorCache::instance = nullptr;
thread_local StatsRecursiveValuesCache *StatsRecursiveValuesCache::instance = nullptr;

/**
 * Helper class to build averages that discared a given amount of outliers.
 * Used for our image variance estimate.
 */
class OutlierRejectedAverage {
public:
    struct Sample {
        Spectrum secondMoment;
        Float cost;

        Sample()
        : secondMoment(Spectrum(0.f)), cost(0) {}

        Sample(const Spectrum &sm, Float cost)
        : secondMoment(sm), cost(cost) {}

        void operator+=(const Sample &other) {
            secondMoment += other.secondMoment;
            cost += other.cost;
        }

        void operator-=(const Sample &other) {
            secondMoment -= other.secondMoment;
            cost -= other.cost;
        }

        Sample operator-(const Sample &other) const {
            Sample s = *this;
            s -= other;
            return s;
        }

        Sample operator/(Float weight) const {
            return Sample {
                secondMoment / weight,
                cost / weight
            };
        }

        bool operator>=(const Sample &other) const {
            return secondMoment.average() >= other.secondMoment.average();
        }
    };

    /**
     * Resizes the history buffer to account for up to \c length outliers.
     */
    void resize(int length) {
        m_length = length;
        m_history.resize(length);
        reset();
    }

    /**
     * Resets all statistics, including outlier history and current average.
     */
    void reset() {
        m_index = 0;
        m_knownMinimum = Sample();
        m_accumulation = Sample();
        m_weight = 0;
        m_outlierAccumulation = Sample();
        m_outlierWeight = 0;
    }

    /**
     * Returns whether a lower bound can be given on what will definitely not count as outlier.
     */
    bool hasOutlierLowerBound() const {
        return m_length > 0 && m_index >= m_length;
    }

    /**
     * Returns the lower bound of what will definitely count as outlier.
     * Useful if multiple \c OutlierRejectedAverage from different threads will be combined.
     */
    Sample outlierLowerBound() const {
        return m_history[m_index - 1];
    }

    /**
     * Sets a manual lower bound of what will count as outlier.
     * This avoids wasting time on adding samples to the outlier history that are known to be less significant
     * than outliers that have already been collected by other instances of \c OutlierRejectedAverage that
     * will eventually be merged.
     */
    void setRemoteOutlierLowerBound(const Sample &minimum) {
        m_knownMinimum = minimum;
    }
    
    /**
     * Records one sample.
     */
    void operator+=(Sample sample) {
        m_weight += 1;
        m_accumulation += sample;

        if (m_knownMinimum >= sample) {
            return;
        }

        int insertionPoint = m_index;
        
        while (insertionPoint > 0 && sample >= m_history[insertionPoint - 1]) {
            if (insertionPoint < m_length) {
                m_history[insertionPoint] = m_history[insertionPoint - 1];
            }
            insertionPoint--;
        }
        
        if (insertionPoint < m_length) {
            m_history[insertionPoint] = sample;
            if (m_index < m_length) {
                ++m_index;
            }
        }
    }
    
    /**
     * Merges the statistics of another \c OutlierRejectedAverage into this instance.
     */
    void operator+=(const OutlierRejectedAverage &other) {
        int m_writeIndex = m_index + other.m_index;
        int m_readIndexLocal = m_index - 1;
        int m_readIndexOther = other.m_index - 1;
        
        while (m_writeIndex > 0) {
            Sample sample;
            if (m_readIndexOther < 0 || (m_readIndexLocal >= 0 && other.m_history[m_readIndexOther] >= m_history[m_readIndexLocal])) {
                /// we take the local sample next
                sample = m_history[m_readIndexLocal--];
            } else {
                /// we take the other sample next
                sample = other.m_history[m_readIndexOther--];
            }
            
            if (--m_writeIndex < m_length) {
                m_history[m_writeIndex] = sample;
            }
        }
        
        m_index = std::min(m_index + other.m_index, m_length);
        m_weight += other.m_weight;
        m_accumulation += other.m_accumulation;
    }
    
    void dump() const {
        std::cout << m_index << " vs " << m_length << std::endl;
        for (int i = 0; i < m_index; ++i)
            std::cout << m_history[i].secondMoment.average() << std::endl;
    }

    void computeOutlierContribution() {
        for (int i = 0; i < m_index; ++i) {
            m_outlierAccumulation += m_history[i];
        }
        m_outlierWeight += m_index;

        /// reset ourselves
        m_index = 0;
    }

    Sample average() const {
        if (m_index > 0) {
            SLog(EWarn, "There are some outliers that have not yet been removed. Did you forget to call computeOutlierContribution()?");
        }

        return (m_accumulation - m_outlierAccumulation) / (m_weight - m_outlierWeight);
    }

    Sample averageWithoutRejection() const {
        return m_accumulation / m_weight;
    }

    long weight() const {
        return m_weight;
    }
    
private:
    long m_weight;
    int m_index;
    int m_length;
    Sample m_accumulation;
    Sample m_knownMinimum;
    std::vector<Sample> m_history;

    Sample m_outlierAccumulation;
    long m_outlierWeight;
};

/**
 * Renders albedo and normals auxiliaries used for denoising the pixel estimate used by ADRRS and our method.
 */
class DenoisingAuxilariesIntegrator : public SamplingIntegrator {
public:
    enum EField {
        EShadingNormal,
        EAlbedo,
    };

    DenoisingAuxilariesIntegrator()
    : SamplingIntegrator(Properties()) {
    }

    Spectrum Li(const RayDifferential &ray, RadianceQueryRecord &rRec) const {
        Spectrum result(0.f);

        if (!rRec.rayIntersect(ray))
            return result;

        Intersection &its = rRec.its;

        switch (m_field) {
            case EShadingNormal:
                result.fromLinearRGB(its.shFrame.n.x, its.shFrame.n.y, its.shFrame.n.z);
                break;
            case EAlbedo:
                result = its.shape->getBSDF()->getDiffuseReflectance(its);
                break;
            default:
                Log(EError, "Internal error!");
        }

        return result;
    }

    std::string toString() const {
        return "DenoisingAuxilariesIntegrator[]";
    }

    EField m_field;
};

class MIRecursivePathTracer : public MonteCarloIntegrator {
private:
    struct RRSMethod {
        enum {
            ENone,
            EClassic,
            EGWTW,
            EADRRS,
            EEARS,
        } technique;

        Float splittingMin;
        Float splittingMax;

        int rrDepth;
        bool useAbsoluteThroughput;

        RRSMethod() {
            technique = ENone;
            splittingMin = 1;
            splittingMax = 1;
            rrDepth = 1;
            useAbsoluteThroughput = true;
        }

        RRSMethod(const Properties &props) {
            /// parse parameters
            splittingMin = props.getFloat("splittingMin", 0.05f);
            splittingMax = props.getFloat("splittingMax", 20);
            rrDepth = props.getInteger("rrDepth", 5);

            /// parse desired modifiers
            std::string rrsStr = props.getString("rrsStrategy", "noRR");
            if ((useAbsoluteThroughput = rrsStr.back() == 'A')) {
                rrsStr.pop_back();
            }

            if (rrsStr.back() == 'S') {
                rrsStr.pop_back();
            } else if (splittingMax > 1) {
                Log(EWarn, "Changing maximum splitting factor to 1 since splitting was not explicitly allowed");
                splittingMax = 1;
            }
            
            /// parse desired technique
            if (rrsStr == "noRR")      technique = ENone; else
            if (rrsStr == "classicRR") technique = EClassic; else
            if (rrsStr == "ADRR")      technique = EADRRS; else
            if (rrsStr == "EAR")       technique = EEARS; else
            if (rrsStr == "GWTWRR")    technique = EGWTW; else {
                Log(EError, "Invalid RRS technique specified: %s", rrsStr.c_str());
            }

            if (technique == EEARS && rrDepth != 1)
                Log(EWarn, "EARS should ideally be used with rrDepth 1");
            
            if (technique == EADRRS && rrDepth != 2)
                Log(EWarn, "ADRRS should ideally be used with rrDepth 2");
        }

        static RRSMethod Classic() {
            RRSMethod rrs;
            rrs.technique    = EClassic;
            rrs.splittingMin = 0;
            rrs.splittingMax = 0.95f;
            rrs.rrDepth      = 5;
            rrs.useAbsoluteThroughput = true;
            return rrs;
        }

        std::string getName() const {
            std::string suffix = "";
            if (splittingMax > 1) suffix += "S";
            if (useAbsoluteThroughput) suffix += "A";

            switch (technique) {
            case ENone:    return "noRR";
            case EClassic: return "classicRR" + suffix;
            case EGWTW:    return "GWTWRR" + suffix;
            case EADRRS:   return "ADRR" + suffix;
            case EEARS:    return "EAR" + suffix;
            default:       return "ERROR";
            }
        }

        Float evaluate(
            const Octtree::SamplingNode *samplingNode,
            Float imageEarsFactor,
            const Spectrum &albedo,
            const Spectrum &throughput,
            Float shininess,
            bool bsdfHasSmoothComponent,
            int depth
        ) const {
            if (depth < rrDepth) {
                /// do not perform RR or splitting at this depth.
                return 1;
            }

            switch (technique) {
            case ENone: {
                /// the simplest mode of all. perform no splitting and no RR.
                return clamp(1);
            }

            case EClassic: {
                /// Classic RR(S) based on throughput weight
                if (albedo.isZero())
                    /// avoid bias for materials that might report their reflectance incorrectly
                    return clamp(0.1f);
                return clamp((throughput * albedo).average());
            }

            case EGWTW: {
                /// "Go with the Winners"
                const Float Vr = 1.0;
                const Float Vv = splittingMax * splittingMax - 1.0;
                return clamp((throughput * albedo).average() * std::sqrt(Vr + Vv / std::pow(shininess + 1, 2)));
            }
            
            case EADRRS: {
                /// "Adjoint-driven Russian Roulette and Splitting"
                const Spectrum LiEstimate = samplingNode->lrEstimate;
                if (bsdfHasSmoothComponent && LiEstimate.max() > 0) {
                    return clamp(weightWindow((throughput * LiEstimate).average()));
                } else {
                    return clamp(1);
                }
            }

            case EEARS: {
                /// "Efficiency-Aware Russian Roulette and Splitting"
                if (bsdfHasSmoothComponent) {
                    const Float splittingFactorS = std::sqrt( (throughput * throughput * samplingNode->earsFactorS).average() ) * imageEarsFactor;
                    const Float splittingFactorR = std::sqrt( (throughput * throughput * samplingNode->earsFactorR).average() ) * imageEarsFactor;

                    if (splittingFactorR > 1) {
                        if (splittingFactorS < 1) {
                            /// second moment and variance disagree on whether to split or RR, resort to doing nothing.
                            return clamp(1);
                        } else {
                            /// use variance only if both modes recommend splitting.
                            return clamp(splittingFactorS);
                        }
                    } else {
                        /// use second moment only if it recommends RR.
                        return clamp(splittingFactorR);
                    }
                } else {
                    return clamp(1);
                }
            }
            }

            /// make gcc happy
            return 0;
        }

        bool needsTrainingPhase() const {
            switch (technique) {
            case ENone:    return false;
            case EClassic: return false;
            case EGWTW:    return false;
            case EADRRS:   return true;
            case EEARS:    return true;
            }

            /// make gcc happy
            return false;
        }

        bool performsInvVarWeighting() const {
            return needsTrainingPhase();
        }
        
        bool needsPixelEstimate() const {
            return useAbsoluteThroughput == false;
        }
    
    private:
        Float clamp(Float splittingFactor) const {
            /// not using std::clamp here since that's C++17
            splittingFactor = std::min(splittingFactor, splittingMax);
            splittingFactor = std::max(splittingFactor, splittingMin);
            return splittingFactor;
        }

        Float weightWindow(Float splittingFactor, Float weightWindowSize = 5) const {
            const float dminus = 2 / (1 + weightWindowSize);
            const float dplus = dminus * weightWindowSize;

            if (splittingFactor < dminus) {
                /// russian roulette
                return splittingFactor / dminus;
            } else if (splittingFactor > dplus) {
                /// splitting
                return splittingFactor / dplus;
            } else {
                /// within weight window
                return 1;
            }
        }
    };

    struct LiInput {
        Spectrum weight;
        Spectrum absoluteWeight; /// only relevant for AOVs
        RayDifferential ray;
        RadianceQueryRecord rRec;
        bool scattered { false };
        Float eta { 1.f };
    };

    struct LiOutput {
        Spectrum reflected { 0.f };
        Spectrum emitted { 0.f };
        Float cost { 0.f };

        int numSamples { 0 };
        Float depthAcc { 0.f };
        Float depthWeight { 0.f };

        void markAsLeaf(int depth) {
            depthAcc = depth;
            depthWeight = 1;
        }

        Float averagePathLength() const {
            return depthWeight > 0 ? depthAcc / depthWeight : 0;
        }

        Float numberOfPaths() const {
            return depthWeight;
        }

        Spectrum totalContribution() const {
            return reflected + emitted;
        }
    };

public:
    /// the cost of ray tracing + direct illumination sample (in seconds)
    static constexpr Float COST_NEE  = 0.3e-7;

    /// the cost of ray tracing + BSDF/camera sample (in seconds)
    static constexpr Float COST_BSDF = 0.3e-7;

    MIRecursivePathTracer(const Properties &props)
    : MonteCarloIntegrator(props) {
        m_oidnDevice = oidn::newDevice();
        m_oidnDevice.commit();
        
        cache.setMaximumMemory(long(24)*1024*1024); /// 24 MiB

        m_renderingRRSMethod = RRSMethod(props);

        m_budget = props.getFloat("budget", 30.0f);
        m_useIncrementalTraining = props.getBoolean("useIncrementalTraining", true);
        m_saveTrainingFrames = props.getBoolean("saveTrainingFrames", false);

        m_outlierRejection = props.getInteger("outlierRejection", 10);
        m_imageStatistics.setOutlierRejectionCount(m_outlierRejection);

        for (const auto &name : props.getPropertyNames()) {
            Log(EInfo, "%s: %s", name.c_str(), props.getAsString(name).c_str());
        }
    }

    ref<BlockedRenderProcess> renderPass(Scene *scene,
        RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {
        
        /* This is a sampling-based integrator - parallelize */
        ref<BlockedRenderProcess> proc = new BlockedRenderProcess(job,
            queue, scene->getBlockSize());

        proc->disableProgress();

        proc->bindResource("integrator", integratorResID);
        proc->bindResource("scene", sceneResID);
        proc->bindResource("sensor", sensorResID);
        proc->bindResource("sampler", samplerResID);

        scene->bindUsedResources(proc);
        bindUsedResources(proc);

        return proc;
    }

    bool renderIterationTime(Float until, int &passesRenderedLocal, Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {
        
        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        Log(EInfo, "ITERATION %d, until %.1f seconds (%s)", m_iteration, until, m_currentRRSMethod.getName().c_str());

        passesRenderedLocal = 0;

        bool result = true;
        while (true) {
            ref<BlockedRenderProcess> process = renderPass(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
            sched->schedule(process);
            sched->wait(process);

            m_imageStatistics.applyOutlierRejection();
            
            ++passesRenderedLocal;
            ++m_passesRenderedGlobal;

            const Float progress = computeElapsedSeconds(m_startTime);
            m_progress->update(progress);
            if (progress > until) {
                break;
            }

            if (process->getReturnStatus() != ParallelProcess::ESuccess) {
                result = false;
                break;
            }
        }

        Log(EInfo, "  %.2f seconds elapsed, passes this iteration: %d, total passes: %d",
            computeElapsedSeconds(m_startTime), passesRenderedLocal, m_passesRenderedGlobal);

        return result;
    }

    static Float computeElapsedSeconds(std::chrono::steady_clock::time_point start) {
        auto current = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
        return (Float)ms.count() / 1000;
    }

    bool renderTime(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        m_progress = std::unique_ptr<ProgressReporter>(new ProgressReporter("Rendering", (int)m_budget, job));
        
        Float iterationTime = 1;
        cache.configuration.leafDecay = m_useIncrementalTraining ? 1 : 0;

        int spp;
        Float until = 0;
        for (m_iteration = 0;; m_iteration++) {
            const Float timeBeforeIter = computeElapsedSeconds(m_startTime);
            if (timeBeforeIter >= m_budget) {
                /// note that we always do at least one sample per pixel per training iteration,
                /// which can sometimes be significantly longer than the budget for that iteration.
                /// this means we can exhaust the training budget before all iterations have finished
                /// (typically due to excessive amounts of splitting)
                break;
            }

            film->clear();
#ifdef EARS_INCLUDE_AOVS
            m_statsImages->clear();
#endif

            /// don't use learning based methods unless caches have somewhat converged
            const bool isPretraining = m_iteration < 3;
            m_currentRRSMethod = isPretraining ? RRSMethod::Classic() : m_renderingRRSMethod;

            until += iterationTime;
            if (until > m_budget - iterationTime) {
                /// since the budget would be exhausted in the next iteration anyway, we exhaust it fully now.
                /// this way we can avoid final iterations that are shorter than "iterationTime" and are not worth the overhead.
                until = m_budget;
            }

            if (!renderIterationTime(until, spp, scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID)) {
                return false;
            }

            updateCaches();
            updateImageStatistics(computeElapsedSeconds(m_startTime) - timeBeforeIter);

            const bool hasVarianceEstimate = m_iteration > 0 || !m_needsPixelEstimate;
            m_finalImage.add(
                film, spp,
                m_renderingRRSMethod.performsInvVarWeighting() ?
                    (hasVarianceEstimate ? m_imageStatistics.squareError().average() : 0) :
                    1
            );

            computePixelEstimate(film);

            if (m_iteration % 8 == 7) {
                /// double the duration of a render pass every 8 passes.
                iterationTime *= 2;
            }

            if (m_saveTrainingFrames) {
                ref<Bitmap> bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat32, film->getSize());
                m_finalImage.develop(bitmap);

                fs::path path = scene->getDestinationFile();
                path = path.parent_path() / (path.leaf().string() + "__train-" + std::to_string(m_iteration) + ".exr");
                Log(EInfo, "Saving training frame to %s", path.c_str());
                bitmap->write(path);
            }
        }

        return true;
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID) {

        ref<Scheduler> sched = Scheduler::getInstance();

        size_t nCores = sched->getCoreCount();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

#ifdef EARS_INCLUDE_AOVS
        auto properties = Properties("hdrfilm");
        properties.setInteger("width", film->getSize().x);
        properties.setInteger("height", film->getSize().y);

        {
            /// debug film with additional channels
            StatsRecursiveDescriptor statsDesc;

            auto properties = Properties(film->getProperties());
            properties.setString("pixelFormat", statsDesc.types);
            properties.setString("channelNames", statsDesc.names);
            std::cout << properties.toString() << std::endl;
            auto rfilter = film->getReconstructionFilter();

            m_debugFilm = static_cast<Film*>(PluginManager::getInstance()->createObject(MTS_CLASS(Film), properties));
            m_debugFilm->addChild(rfilter);
            m_debugFilm->configure();

            m_statsImages.reset(new StatsRecursiveImageBlocks([&]() {
                return new ImageBlock(Bitmap::ESpectrumAlphaWeight, film->getCropSize());
            }));
            m_debugImage = new ImageBlock(Bitmap::EMultiSpectrumAlphaWeight, film->getCropSize(), NULL,
                statsDesc.size * SPECTRUM_SAMPLES + 2
            );
        }
#endif

        m_needsPixelEstimate = m_renderingRRSMethod.needsPixelEstimate();
        m_needsCaches = m_renderingRRSMethod.needsTrainingPhase();
        renderDenoisingAuxiliaries(scene, queue, job, sceneResID, sensorResID);

        m_startTime = std::chrono::steady_clock::now();

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y, nCores, nCores == 1 ? "core" : "cores");

        Thread::initializeOpenMP(nCores);

        int integratorResID = sched->registerResource(this);
        bool result = true;

        m_passesRenderedGlobal = 0;
        m_finalImage.clear();

        result = renderTime(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);

        Vector2i size = film->getSize();
        ref<Bitmap> image = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, size);
        film->develop(Point2i(0, 0), size, Point2i(0, 0), image);

#ifdef EARS_INCLUDE_AOVS
        auto statsBitmaps = m_statsImages->getBitmaps();
        Float* debugImage = m_debugImage->getBitmap()->getFloatData();

        for (int y = 0; y < size.y; ++y)
            for (int x = 0; x < size.x; ++x) {
                Point2i pos = Point2i(x, y);
                Spectrum pixel = image->getPixel(pos);

                /// write out debug channels
                for (int i = 0; i < SPECTRUM_SAMPLES; ++i) *(debugImage++) = pixel[i];

                for (auto &b : statsBitmaps) {
                    Spectrum v = b->getPixel(pos);
                    for (int i = 0; i < SPECTRUM_SAMPLES; ++i) *(debugImage++) = v[i];
                }
                
                *(debugImage++) = 1.0f;
                *(debugImage++) = 1.0f;
            }

        m_debugFilm->setBitmap(m_debugImage->getBitmap());

        {
            /// output debug image
            std::string suffix = "-dbg-" + m_renderingRRSMethod.getName() + "-" + std::to_string(m_passesRenderedGlobal) + "spp";
            fs::path destPath = scene->getDestinationFile();
            fs::path debugPath = destPath.parent_path() / (
                destPath.leaf().string()
                + suffix
                + ".exr"
            );

            m_debugFilm->setDestinationFile(debugPath, 0);
            m_debugFilm->develop(scene, 0.0f);
        }
#endif

        ref<Bitmap> finalBitmap = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, film->getSize());
        m_finalImage.develop(finalBitmap.get());
        film->setBitmap(finalBitmap);

        sched->unregisterResource(integratorResID);
        m_progress = nullptr;

        return result;
    }

    void renderBlock(const Scene *scene, const Sensor *sensor,
        Sampler *sampler, ImageBlock *block, const bool &stop,
        const std::vector< TPoint2<uint8_t> > &points) const {

        static thread_local OutlierRejectedAverage blockStatistics;
        blockStatistics.resize(m_outlierRejection);
        if (m_imageStatistics.hasOutlierLowerBound()) {
            blockStatistics.setRemoteOutlierLowerBound(m_imageStatistics.outlierLowerBound());
        }

        bool needsApertureSample = sensor->needsApertureSample();
        bool needsTimeSample = sensor->needsTimeSample();

        RadianceQueryRecord rRec(scene, sampler);
        Point2 apertureSample(0.5f);
        Float timeSample = 0.5f;
        RayDifferential sensorRay;

        block->clear();

#ifdef EARS_INCLUDE_AOVS
        static thread_local StatsRecursiveImageBlocks blocks([&]() {
            auto b = new ImageBlock(block->getPixelFormat(), block->getSize(), block->getReconstructionFilter());
            return b;
        });

        for (auto &b : blocks.blocks) {
            b->setOffset(block->getOffset());
            b->clear();
        }
#endif

        StatsRecursiveValues stats;

        uint32_t queryType = RadianceQueryRecord::ESensorRay;

        if (!sensor->getFilm()->hasAlpha()) // Don't compute an alpha channel if we don't have to
            queryType &= ~RadianceQueryRecord::EOpacity;

        Float depthAcc = 0;
        Float depthWeight = 0;
        Float primarySplit = 0;
        Float samplesTaken = 0;

        for (size_t i = 0; i < points.size(); ++i) {
            Point2i offset = Point2i(points[i]) + Vector2i(block->getOffset());
            //if (stop)
            //    break;

            Spectrum pixelEstimate { 0.5f };
            if (m_pixelEstimate.get()) {
                pixelEstimate = m_pixelEstimate->getPixel(offset);
            }

            const Spectrum metricNorm = m_renderingRRSMethod.useAbsoluteThroughput ?
                Spectrum { 1.f } :
                pixelEstimate + Spectrum { 1e-2 };
            const Spectrum expectedContribution = pixelEstimate / metricNorm;

            constexpr int sppPerPass = 1;
            for (int j = 0; j < sppPerPass; j++) {
                stats.reset();

                rRec.newQuery(queryType, sensor->getMedium());
                Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));

                if (needsApertureSample)
                    apertureSample = rRec.nextSample2D();
                if (needsTimeSample)
                    timeSample = rRec.nextSample1D();

                Spectrum spec = sensor->sampleRayDifferential(
                    sensorRay, samplePos, apertureSample, timeSample);

                LiInput input;
                input.absoluteWeight = spec;
                input.weight = spec;
                input.ray = sensorRay;
                input.rRec = rRec;
                if (!m_currentRRSMethod.useAbsoluteThroughput)
                    input.weight /= pixelEstimate + Spectrum { 1e-2 };

                LiOutput output = Li(input, stats);
                block->put(samplePos, spec * output.totalContribution(), input.rRec.alpha);
                sampler->advance();

                const Spectrum pixelContribution = (spec / metricNorm) * output.totalContribution();
                const Spectrum diff = pixelContribution - expectedContribution;

                blockStatistics += OutlierRejectedAverage::Sample {
                    diff * diff,
                    output.cost
                };
                
                stats.pixelEstimate.add(pixelEstimate);
                stats.avgPathLength.add(output.averagePathLength());
                stats.numPaths.add(output.numberOfPaths());
                stats.cost.add(1e+6 * output.cost);

#ifdef EARS_INCLUDE_AOVS
                stats.put(blocks, samplePos, rRec.alpha);
#endif

                depthAcc += output.depthAcc;
                depthWeight += output.depthWeight;
                primarySplit += output.numSamples;
                samplesTaken += 1;
            }
        }

        //if (!stop) {
#ifdef EARS_INCLUDE_AOVS
            m_statsImages->put(blocks);
#endif
            m_imageStatistics += blockStatistics;
            m_imageStatistics.splatDepthAcc(depthAcc, depthWeight, primarySplit, samplesTaken);
        //}
    }

    Vector3f mapPointToUnitCube(const Scene *scene, const Point3 &point) const {
        AABB aabb = scene->getAABB();
        Vector3f size = aabb.getExtents();
        Vector3f result = point - aabb.min;
        for (int i = 0; i < 3; ++i)
            result[i] /= size[i];
        return result;
    }

    Point2 dirToCanonical(const Vector& d) const {
        if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.z)) {
            return {0, 0};
        }

        const Float cosTheta = std::min(std::max(d.z, -1.0f), 1.0f);
        Float phi = std::atan2(d.y, d.x);
        while (phi < 0)
            phi += 2.0 * M_PI;

        return {(cosTheta + 1) / 2, phi / (2 * M_PI)};
    }

    int mapOutgoingDirectionToHistogramBin(const Vector3f &wo) const {
        const Point2 p = dirToCanonical(wo);
        const int res = Octtree::HISTOGRAM_RESOLUTION;
        const int result =
            std::min(int(p.x * res), res - 1) +
            std::min(int(p.y * res), res - 1) * res;
        return result;
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        Assert(false);
        return Spectrum { 0.f };
    }

    LiOutput Li(LiInput &input, StatsRecursiveValues &stats) const {
        LiOutput output;

        if (m_maxDepth >= 0 && input.rRec.depth > m_maxDepth) {
            // maximum depth reached
            output.markAsLeaf(input.rRec.depth);
            return output;
        }

        /* Some aliases and local variables */
        RadianceQueryRecord &rRec = input.rRec;
        Intersection &its = rRec.its;
        const Scene *scene = rRec.scene;
        RayDifferential ray(input.ray);

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        if (rRec.type & RadianceQueryRecord::EIntersection) {
            rRec.rayIntersect(ray);
            output.cost += COST_BSDF;
        }

        if (!its.isValid()) {
            /* If no intersection could be found, potentially return
                radiance from a environment luminaire if it exists */
            if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (!m_hideEmitters || input.scattered))
                output.emitted += scene->evalEnvironment(ray);
            stats.emitted.add(rRec.depth-1, input.absoluteWeight * output.emitted, 0);
            output.markAsLeaf(rRec.depth);
            return output;
        }

        const BSDF *bsdf = its.getBSDF();
        const bool bsdfHasSmoothComponent = bsdf->getType() & BSDF::ESmooth;

        /* Possibly include emitted radiance if requested */
        if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
            && (!m_hideEmitters || input.scattered))
            output.emitted += its.Le(-ray.d);

        /* Include radiance from a subsurface scattering model if requested */
        if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
            output.emitted += its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);
        
        stats.emitted.add(rRec.depth-1, input.absoluteWeight * output.emitted, 0);

        const Float wiDotGeoN = -dot(its.geoFrame.n, ray.d);
        const Float wiDotShN = Frame::cosTheta(its.wi);
        if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
            || (m_strictNormals && wiDotGeoN * wiDotShN < 0)) {

            /* Only continue if:
                1. The current path length is below the specifed maximum
                2. If 'strictNormals'=true, when the geometric and shading
                    normals classify the incident direction to the same side */
            output.markAsLeaf(rRec.depth);
            return output;
        }

        /* ==================================================================== */
        /*                 Compute reflected radiance estimate                  */
        /* ==================================================================== */

        /// fetch some information about the BSDF
        const Spectrum albedo = bsdf->getDiffuseReflectance(its) + bsdf->getSpecularReflectance(its);
        Float roughness = std::numeric_limits<Float>::infinity();
        for (int comp = 0; comp < bsdf->getComponentCount(); ++comp) {
            roughness = std::min(roughness, bsdf->getRoughness(its, comp));
        }
        /// we use a simple approximation to convert (assumed) beckmann-roughness to shininess
        const Float shininess = std::max(0.f, 2.f / (roughness * roughness) - 2.f);

        /// look up the intersection point in our spatial cache
        const int histogramBinIndex = mapOutgoingDirectionToHistogramBin(input.ray.d);
        const Octtree::SamplingNode *samplingNode = nullptr;
        Octtree::TrainingNode *trainingNode = nullptr;

        if (m_needsCaches) {
            cache.lookup(mapPointToUnitCube(scene, its.p), histogramBinIndex, samplingNode, trainingNode);
        }

        /// update AOVs
        if (rRec.depth == 1) {
            stats.albedo.add(albedo);
            stats.roughness.add(roughness);
            
            if (samplingNode) {
                stats.earsFactorS.add(samplingNode->earsFactorS * m_imageEarsFactor);
                stats.earsFactorR.add(samplingNode->earsFactorR * m_imageEarsFactor);
                stats.lrEstimate.add(samplingNode->lrEstimate);
            }
        }

        /// compute splitting factor
        const Float splittingFactor = m_currentRRSMethod.evaluate(
            samplingNode, m_imageEarsFactor,
            albedo, input.weight, shininess, bsdfHasSmoothComponent,
            rRec.depth
        );
        stats.splittingFactor.add(rRec.depth-1, splittingFactor);

        Spectrum lrSum(0.f);
        Spectrum lrSumSquares(0.f);
        Float lrSumCosts = 0.f;

        /// actual number of samples is the stochastic rounding of our splittingFactor
        const int numSamples = int(splittingFactor + rRec.nextSample1D());
        output.numSamples = numSamples;
        for (int sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex) {
            Spectrum irradianceEstimate(0.f);
            Spectrum LrEstimate(0.f);
            Float LrCost(0.f);

            /* ==================================================================== */
            /*                     Direct illumination sampling                     */
            /* ==================================================================== */

            DirectSamplingRecord dRec(its);

            /* Estimate the direct illumination if this is requested */
            if ((rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) &&
                (bsdf->getType() & BSDF::ESmooth)) {
                LrCost += COST_NEE;

                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    Spectrum bsdfVal = bsdf->eval(bRec);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!bsdfVal.isZero() && (!m_strictNormals
                        || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

                        /* Calculate prob. of having generated that direction
                            using BSDF sampling */
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                            ? bsdf->pdf(bRec) : 0;

                        /* Weight using the power heuristic */
                        Float misWeight = miWeight(dRec.pdf, bsdfPdf);
                        Float absCosTheta = std::abs(Frame::cosTheta(bRec.wo));

                        LrEstimate += bsdfVal * value * misWeight;
                        irradianceEstimate += absCosTheta * value * misWeight;

                        stats.emitted.add(rRec.depth, input.absoluteWeight * bsdfVal * value * misWeight / splittingFactor, 0);
                    }
                }
            }
            
            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

            Spectrum bsdfWeight(0.f);
            Float bsdfPdf;
            Spectrum LiEstimate(0.f);

            do {
                LiInput inputNested  = input;
                inputNested.weight *= 1.f / splittingFactor;
                inputNested.absoluteWeight *= 1.f / splittingFactor;
                inputNested.rRec.its = rRec.its;

                RadianceQueryRecord &rRec = inputNested.rRec;
                Intersection &its = rRec.its;
                RayDifferential &ray = inputNested.ray;

                /* Sample BSDF * cos(theta) */
                BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
                bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
                if (bsdfWeight.isZero())
                    break;

                Float absCosTheta = std::abs(Frame::cosTheta(bRec.wo));
                inputNested.scattered |= bRec.sampledType != BSDF::ENull;

                /* Prevent light leaks due to the use of shading normals */
                const Vector wo = its.toWorld(bRec.wo);
                Float woDotGeoN = dot(its.geoFrame.n, wo);
                if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
                    break;

                bool hitEmitter = false;
                Spectrum value;

                /* Trace a ray in this direction */
                ray = Ray(its.p, wo, ray.time);
                LrCost += COST_BSDF;
                if (scene->rayIntersect(ray, its)) {
                    /* Intersected something - check if it was a luminaire */
                    if (its.isEmitter()) {
                        value = its.Le(-ray.d);
                        dRec.setQuery(ray, its);
                        hitEmitter = true;
                    }
                } else {
                    /* Intersected nothing -- perhaps there is an environment map? */
                    const Emitter *env = scene->getEnvironmentEmitter();

                    if (env) {
                        if (m_hideEmitters && !inputNested.scattered)
                            break;

                        value = env->evalEnvironment(ray);
                        if (!env->fillDirectSamplingRecord(dRec, ray))
                            break;
                        hitEmitter = true;
                    } else {
                        break;
                    }
                }

                /* Keep track of the throughput, medium, and relative
                refractive index along the path */
                inputNested.weight *= bsdfWeight;
                inputNested.absoluteWeight *= bsdfWeight;
                inputNested.eta *= bRec.eta;

                /* If a luminaire was hit, estimate the local illumination and
                    weight using the power heuristic */
                if (hitEmitter &&
                    (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                    /* Compute the prob. of generating that direction using the
                        implemented direct illumination sampling technique */
                    Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ? scene->pdfEmitterDirect(dRec) : 0;
                    Float misWeight = miWeight(bsdfPdf, lumPdf);
                    LrEstimate += bsdfWeight * value * misWeight;
                    irradianceEstimate += absCosTheta * (value / bsdfPdf) * misWeight;
                    stats.emitted.add(rRec.depth, inputNested.absoluteWeight * value * misWeight, 0);
                }

                /* ==================================================================== */
                /*                         Indirect illumination                        */
                /* ==================================================================== */

                /* Set the recursive query type. Stop if no surface was hit by the
                    BSDF sample or if indirect illumination was not requested */
                if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                    break;
                rRec.type = RadianceQueryRecord::ERadianceNoEmission & ~RadianceQueryRecord::EIntersection;
                rRec.depth++;

                LiOutput outputNested = this->Li(inputNested, stats);
                LrEstimate += bsdfWeight * outputNested.totalContribution();
                irradianceEstimate += absCosTheta * (outputNested.totalContribution() / bsdfPdf);
                LrCost += outputNested.cost;

                output.depthAcc += outputNested.depthAcc;
                output.depthWeight += outputNested.depthWeight;
            } while (false);

            output.reflected += LrEstimate / splittingFactor;
            output.cost += LrCost;

            if (m_needsCaches) {
                lrSum += LrEstimate;
                lrSumSquares += LrEstimate * LrEstimate;
                lrSumCosts += LrCost;
            }
        }

        if (m_needsCaches && numSamples > 0) {
            trainingNode->splatLrEstimate(
                lrSum,
                lrSumSquares,
                lrSumCosts,
                numSamples
            );
        }

        if (output.depthAcc == 0) {
            /// all BSDF samples have failed :-(
            output.markAsLeaf(rRec.depth);
        }

        return output;
    }

    inline Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "MIRecursivePathTracer[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

    bool renderDenoisingAuxiliaries(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID) {
        
        if (!m_needsPixelEstimate) {
            Log(EInfo, "Not rendering denoising auxilaries as they are not required by '%s'",
                m_renderingRRSMethod.getName().c_str());
            return true;
        }

        Log(EInfo, "Rendering auxiliaries for pixel estimates");

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        Properties samplerProperties { "ldsampler" };
        samplerProperties.setInteger("sampleCount", 128);
        
        ref<Sampler> sampler = static_cast<Sampler *>(PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), samplerProperties));

        std::vector<SerializableObject *> samplers(sched->getCoreCount());
        for (size_t i=0; i<sched->getCoreCount(); ++i) {
            ref<Sampler> clonedSampler = sampler->clone();
            clonedSampler->incRef();
            samplers[i] = clonedSampler.get();
        }

        int samplerResID = sched->registerMultiResource(samplers);
        for (size_t i=0; i<sched->getCoreCount(); ++i)
            samplers[i]->decRef();
        
        ref<DenoisingAuxilariesIntegrator> integrator = new DenoisingAuxilariesIntegrator();

        bool result = true;

        /// render normals
        film->clear();
        integrator->m_field = DenoisingAuxilariesIntegrator::EField::EShadingNormal;
        result &= integrator->render(scene, queue, job, sceneResID, sensorResID, samplerResID);
        m_denoiseAuxNormals = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, film->getSize());
        film->develop(Point2i(0, 0), film->getSize(), Point2i(0, 0), m_denoiseAuxNormals);

        /// render albedo
        film->clear();
        integrator->m_field = DenoisingAuxilariesIntegrator::EField::EAlbedo;
        result &= integrator->render(scene, queue, job, sceneResID, sensorResID, samplerResID);
        m_denoiseAuxAlbedo = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, film->getSize());
        film->develop(Point2i(0, 0), film->getSize(), Point2i(0, 0), m_denoiseAuxAlbedo);

        sched->unregisterResource(samplerResID);

        return result;
    }

    void updateCaches() {
        if (m_needsCaches) {
            cache.build(true);
        }
    }

    void updateImageStatistics(Float actualTotalCost) {
        m_imageStatistics.reset(actualTotalCost);
        m_imageEarsFactor = m_imageStatistics.earsFactor();
    }

    void computePixelEstimate(const ref<Film> &film) {
        if (!m_needsPixelEstimate)
            return;
        
        const Vector2i size = film->getSize();
        if (!m_pixelEstimate) {
            m_pixelEstimate = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, size);
        }

        const int bytePixelStride = m_pixelEstimate->getBytesPerPixel();
        const int byteRowStride = size.x * bytePixelStride;

        if (m_finalImage.hasData()) {
            m_finalImage.develop(m_pixelEstimate.get());
        } else {
            film->develop(Point2i(0, 0), size, Point2i(0, 0), m_pixelEstimate);
        }
        
        auto filter = m_oidnDevice.newFilter("RT");
        filter.setImage("color", m_pixelEstimate->getData(), oidn::Format::Float3, size.x, size.y, 0, bytePixelStride, byteRowStride);
        filter.setImage("albedo", m_denoiseAuxAlbedo->getData(), oidn::Format::Float3, size.x, size.y, 0, bytePixelStride, byteRowStride);
        filter.setImage("normal", m_denoiseAuxNormals->getData(), oidn::Format::Float3, size.x, size.y, 0, bytePixelStride, byteRowStride);
        filter.setImage("output", m_pixelEstimate->getData(), oidn::Format::Float3, size.x, size.y, 0, bytePixelStride, byteRowStride);
        filter.set("hdr", true);
        filter.commit();
        filter.execute();
        
        const char *error;
        if (m_oidnDevice.getError(error) != oidn::Error::None) {
            Log(EError, "OpenImageDenoise: %s", error);
        } else {
            Log(EInfo, "OpenImageDenoise finished successfully");
        }
    }

private:
    mutable Octtree cache;

    std::unique_ptr<StatsRecursiveImageBlocks> m_statsImages;
    mutable ref<ImageBlock> m_debugImage;
    mutable ref<Film> m_debugFilm;

    struct WeightedBitmapAccumulator {
        void clear() {
            m_scrap = nullptr;
            m_bitmap = nullptr;
            m_spp = 0;
            m_weight = 0;
        }

        bool hasData() const {
            return m_weight > 0;
        }

        void add(const ref<Film> &film, int spp, Float avgVariance = 1) {
            if (avgVariance == 0 && m_weight > 0) {
                SLog(EError, "Cannot add an image with unknown variance to an already populated accumulator");
                return;
            }

            const Vector2i size = film->getSize();
            const long floatCount = size.x * size.y * long(SPECTRUM_SAMPLES);

            if (!m_scrap) {
                m_scrap = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, size);
            }
            film->develop(Point2i(0, 0), size, Point2i(0, 0), m_scrap);
            
            ///

            if (!m_bitmap) {
                m_bitmap = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, size);

                Float *m_bitmapData = m_bitmap->getFloat32Data();
                for (long i = 0; i < floatCount; ++i) {
                    m_bitmapData[i] = 0;
                }
            }

            Float *m_bitmapData = m_bitmap->getFloat32Data();
            if (avgVariance > 0 && m_weight == 0 && m_spp > 0) {
                /// reweight previous frames that had unknown variance with our current variance estimate
                const Float reweight = 1 / avgVariance;
                for (long i = 0; i < floatCount; ++i) {
                    m_bitmapData[i] *= reweight;
                }
                m_weight += m_spp * reweight;
            }

            const Float weight = avgVariance > 0 ? spp / avgVariance : spp;
            const Float *m_scrapData = m_scrap->getFloat32Data();
            for (long i = 0; i < floatCount; ++i) {
                m_bitmapData[i] += m_scrapData[i] * weight;
            }

            m_weight += avgVariance > 0 ? weight : 0;
            m_spp += spp;
        }

        void develop(Bitmap *dest) const {
            if (!m_bitmap) {
                SLog(EWarn, "Cannot develop bitmap, as no data is available");
                return;
            }

            const Vector2i size = m_bitmap->getSize();
            const long floatCount = size.x * size.y * long(SPECTRUM_SAMPLES);

            const Float weight = m_weight == 0 ? m_spp : m_weight;
            Float *m_destData = dest->getFloat32Data();
            const Float *m_bitmapData = m_bitmap->getFloat32Data();
            for (long i = 0; i < floatCount; ++i) {
                m_destData[i] = weight > 0 ? m_bitmapData[i] / weight : 0;
            }
        }

        void develop(const fs::path &path) const {
            if (!m_scrap) {
                SLog(EWarn, "Cannot develop bitmap, as no data is available");
                return;
            }

            develop(m_scrap.get());
            m_scrap->write(path);
        }

    private:
        mutable ref<Bitmap> m_scrap;
        ref<Bitmap> m_bitmap;
        Float m_weight;
        int m_spp;
    } m_finalImage;

    RRSMethod m_currentRRSMethod;
    RRSMethod m_renderingRRSMethod;

    int m_iteration;
    int m_passesRenderedGlobal;

    bool m_useIncrementalTraining;
    bool m_saveTrainingFrames;
    Float m_budget;
    int m_outlierRejection;

    mutable std::unique_ptr<ProgressReporter> m_progress;
    std::chrono::steady_clock::time_point m_startTime;

    mutable struct ImageStatistics {
        void setOutlierRejectionCount(int count) {
            m_average.resize(count);
        }

        void applyOutlierRejection() {
            m_average.computeOutlierContribution();
        }

        Float cost() const {
            return m_lastStats.cost;
        }

        Spectrum squareError() const {
            return m_lastStats.squareError;
        }

        Float earsFactor() const {
            return std::sqrt( cost() / squareError().average() );
        }

        Float efficiency() const {
            return 1 / (cost() * squareError().average());
        }

        void reset(Float actualTotalCost) {
            auto weight = m_average.weight();
            auto avgNoReject = m_average.averageWithoutRejection();
            auto avg = m_average.average();
            m_lastStats.squareError = avg.secondMoment;
            m_lastStats.cost = avg.cost;

            //m_average.dump();
            m_average.reset();

            Float earsFactorNoReject = std::sqrt( avgNoReject.cost / avgNoReject.secondMoment.average() );

            Log(EInfo, "Average path count:  %.3f", m_primarySamples > 0 ? m_depthWeight / m_primarySamples : 0);
            Log(EInfo, "Average path length: %.3f", m_depthWeight > 0 ? m_depthAcc / m_depthWeight : 0);
            Log(EInfo, "Average primary split: %.3f", m_primarySamples > 0 ? m_primarySplit / m_primarySamples : 0);
            Log(EInfo, "Statistics:\n"
                "  (values in brackets are without outlier rejection)\n"
                "  Estimated Cost    = %.3e (%.3e)\n"
                "  Actual Cost       = %.3e (  n. a.  )\n"
                "  Variance per SPP  = %.3e (%.3e)\n"
                "  Est. Cost per SPP = %.3e (%.3e)\n"
                "  Est. Efficiency   = %.3e (%.3e)\n"
                "  Act. Efficiency   = %.3e (%.3e)\n"
                "  EARS multiplier   = %.3e (%.3e)\n",
                avg.cost * weight, avgNoReject.cost * weight,
                actualTotalCost,
                squareError().average(), avgNoReject.secondMoment.average(),
                cost(), avgNoReject.cost,
                efficiency(), 1 / (avgNoReject.cost * avgNoReject.secondMoment.average()),
                1 / (actualTotalCost / weight * squareError().average()), 1 / (actualTotalCost / weight * avgNoReject.secondMoment.average()),
                earsFactor(), earsFactorNoReject
            );

            m_depthAcc = 0;
            m_depthWeight = 0;
            m_primarySplit = 0;
            m_primarySamples = 0;
        }

        void operator+=(const OutlierRejectedAverage &blockStatistics) {
            std::lock_guard<std::mutex> lock(m_averageMutex);
            m_average += blockStatistics;
        }

        void splatDepthAcc(Float depthAcc, Float depthWeight, Float primarySplit, Float primarySamples) {
            atomicAdd(&m_depthAcc, depthAcc);
            atomicAdd(&m_depthWeight, depthWeight);
            atomicAdd(&m_primarySplit, primarySplit);
            atomicAdd(&m_primarySamples, primarySamples);
        }

        bool hasOutlierLowerBound() const {
            return m_average.hasOutlierLowerBound();
        }

        OutlierRejectedAverage::Sample outlierLowerBound() const {
            return m_average.outlierLowerBound();
        }

    private:
        std::mutex m_averageMutex;
        OutlierRejectedAverage m_average;
        Float m_depthAcc { 0.f };
        Float m_depthWeight { 0.f };
        Float m_primarySplit { 0.f };
        Float m_primarySamples { 0.f };

        struct {
            Spectrum squareError;
            Float cost;
        } m_lastStats;
    } m_imageStatistics;

    Float m_imageEarsFactor;

    bool m_needsPixelEstimate;
    bool m_needsCaches;
    oidn::DeviceRef m_oidnDevice;
    ref<Bitmap> m_pixelEstimate;
    ref<Bitmap> m_denoiseAuxNormals;
    ref<Bitmap> m_denoiseAuxAlbedo;

public:
    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS(MIRecursivePathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(MIRecursivePathTracer, "MI recursive path tracer");
MTS_NAMESPACE_END
