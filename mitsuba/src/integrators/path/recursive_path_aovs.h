#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/scene.h>

#include <string>
#include <vector>

MTS_NAMESPACE_BEGIN

template<template<typename Data> class Entry>
struct StatsRecursive {
    template<typename T, int Count>
    struct Stack {
#ifdef EARS_INCLUDE_AOVS
        std::unique_ptr<Entry<T>> first;
        std::unique_ptr<Entry<T>> entries[Count];

        int minDepth = -1;

        Stack(const std::string &name) {
            first.reset(new Entry<T>(name));
            for (int i = 0; i < Count; ++i) {
                entries[i].reset(new Entry<T>(name + "." + std::to_string(i)));
            }
        }

        void reset() {
            minDepth = -1;
            first->reset();
            for (int i = 0; i < Count; ++i)
                entries[i]->reset();
        }

        void add(int depth, const T &value, Float weight = 1) {
            SAssert(depth >= 0);

            if constexpr (Count > 0) {
                if (depth < Count)
                    entries[depth]->add(value, weight);
            }
            
            if (depth <= minDepth || minDepth == -1) {
                if (depth < minDepth)
                    first->reset();
                minDepth = depth;
                first->add(value, weight);
            }
        }
#else
    Stack(const std::string &name) {}
    void reset() {}
    void add(int depth, const T &value, Float weight = 1) {}
#endif
    };

    Entry<Spectrum>    albedo          { "pixel.albedo"    };
    Entry<Float>       roughness       { "pixel.roughness" };
    Entry<Spectrum>    pixelEstimate   { "pixel.estimate"  };
    Entry<Float>       cost            { "pixel.cost"      };
    Entry<Float>       avgPathLength   { "paths.length"    };
    Entry<Float>       numPaths        { "paths.count"     };
    Stack<Float, 4>    splittingFactor { "d.rrs"           };
    Entry<Spectrum>    earsFactorS     { "d.ears.s"        };
    Entry<Spectrum>    earsFactorR     { "d.ears.r"        };
    Entry<Spectrum>    lrEstimate      { "d.lrEstimate"    };
    Stack<Spectrum, 6> emitted         { "e.emitted"       };

    void reset() {
        albedo.reset();
        pixelEstimate.reset();
        cost.reset();

        avgPathLength.reset();
        numPaths.reset();

        splittingFactor.reset(); 
        roughness.reset();
        emitted.reset();
        lrEstimate.reset();

        earsFactorS.reset();
        earsFactorR.reset();
    }
};

template<typename T>
struct FormatDescriptor {};

template<>
struct FormatDescriptor<Float> {
    int numComponents = 1;
    Bitmap::EPixelFormat pixelFormat = Bitmap::EPixelFormat::ELuminance;
    std::string pixelName = "luminance";
};

template<>
struct FormatDescriptor<Spectrum> {
    int numComponents = SPECTRUM_SAMPLES;
    Bitmap::EPixelFormat pixelFormat = Bitmap::EPixelFormat::ESpectrum;
    std::string pixelName = "rgb";
};

struct StatsRecursiveImageBlockCache {
    thread_local static StatsRecursiveImageBlockCache *instance;
    StatsRecursiveImageBlockCache(std::function<ImageBlock *()> createImage)
    : createImage(createImage) {
        instance = this;
    }

    std::function<ImageBlock *()> createImage;
    mutable std::vector<ref<ImageBlock>> blocks;
};

template<typename T>
struct StatsRecursiveImageBlockEntry {
    StatsRecursiveImageBlockEntry(const std::string &) {
        image = StatsRecursiveImageBlockCache::instance->createImage();
        image->setWarn(false); // some statistics can be negative
        StatsRecursiveImageBlockCache::instance->blocks.push_back(image);
    }

    ImageBlock *image;

    void add(const T &, Float) {}
};

struct StatsRecursiveImageBlocks : StatsRecursiveImageBlockCache, StatsRecursive<StatsRecursiveImageBlockEntry> {
    StatsRecursiveImageBlocks(std::function<ImageBlock *()> createImage)
    : StatsRecursiveImageBlockCache(createImage) {}

    void clear() {
        for (auto &block : blocks)
            block->clear();
    }

    void put(StatsRecursiveImageBlocks &other) const {
        for (size_t i = 0; i < blocks.size(); ++i) {
            blocks[i]->put(other.blocks[i]);
        }
    }

    std::vector<Bitmap *> getBitmaps() {
        std::vector<Bitmap *> result;
        for (auto &block : blocks)
            result.push_back(block->getBitmap());
        return result;
    }
};

struct StatsRecursiveDescriptorCache {
    thread_local static StatsRecursiveDescriptorCache *instance;
    StatsRecursiveDescriptorCache() {
        instance = this;
    }

    std::string names = "color", types = "rgb";

    int size = 1;
    int components = SPECTRUM_SAMPLES;
};

template<typename T>
struct StatsRecursiveDescriptorEntry {
    StatsRecursiveDescriptorEntry(const std::string &name) {
        auto &cache = *StatsRecursiveDescriptorCache::instance;

        cache.names += ", " + name;

        FormatDescriptor<T> fmt;
        cache.components += fmt.numComponents;
        cache.types += ", " + fmt.pixelName;
        
        cache.size += 1;
    }

    void add(const T &, Float) {}
};

struct StatsRecursiveDescriptor : StatsRecursiveDescriptorCache, StatsRecursive<StatsRecursiveDescriptorEntry> {
};

struct StatsRecursiveValuesCache {
    thread_local static StatsRecursiveValuesCache *instance;
    StatsRecursiveValuesCache() {
        instance = this;
    }

    std::vector<std::function<void (ImageBlock *, const Point2 &, Float)>> putters;
};

template<typename T>
struct StatsRecursiveValueEntry {
    StatsRecursiveValueEntry(const std::string &) {
        StatsRecursiveValuesCache::instance->putters.push_back([&](ImageBlock *block, const Point2 &samplePos, Float alpha) {
            Spectrum spec { value };
            Float temp[SPECTRUM_SAMPLES + 2];
            for (int i = 0; i < SPECTRUM_SAMPLES; ++i)
                temp[i] = spec[i];
            temp[SPECTRUM_SAMPLES] = 1.0f;
            temp[SPECTRUM_SAMPLES + 1] = weight > 0 ? weight : 1;
            block->put(samplePos, temp);
        });
    }

    T value { 0.f };
    Float weight = 0.f;

    void reset() {
        value = T { 0.f };
        weight = 0.f;
    }

    void increment() {
        value++;
    }

    void add(const T &v, Float w = 1) {
        value += v;
        weight += w;
    }
};

struct StatsRecursiveValues : StatsRecursiveValuesCache, StatsRecursive<StatsRecursiveValueEntry> {
    void put(StatsRecursiveImageBlocks &other, const Point2 &samplePos, Float alpha) {
        for (size_t i = 0; i < putters.size(); ++i)
            putters[i](other.blocks[i], samplePos, alpha);
    }
};

MTS_NAMESPACE_END
