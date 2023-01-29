/*
    A genetic algorithm that uses:
        1. the backpack problem to generate the fitness of the population
        2. the (russian) roulette algorithm for chromosome selection
        3. the uniform binary crossover algorithm for chromosome crossing
        4. the hard binary mutation for chromosome mutation
*/

#include <functional>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cassert>
#include <ranges>
#include <random>

using namespace std;

template<typename Object, typename Iterable>
void Print(
    const Iterable& iterable,
    const string& separatorDimensions = "\n",
    const function<void(const Object&)>& funcPrintElem = [] (const Object& obj) {
        static_assert(
            is_arithmetic_v<Object> || is_same_v<remove_const_t<remove_pointer_t<Object>>, char>,
            R"(The object from the innermost range is not a built-in/c-string type, please provide a valid print element function.)"
            );
        cout << obj << ' ';
    }
) {
    if constexpr (ranges::range<Iterable>) {
        ranges::for_each(iterable, [&] (const auto& it) { Print(it, separatorDimensions, funcPrintElem); });
        cout << separatorDimensions;
    } else {
        funcPrintElem(iterable);
    }
}

template<typename T>
using matrix = vector<vector<T>>;

class Random {
public:
    Random() : mGenerator(mSeeder()) {}

    template <typename Type>
    inline auto Get(
        const Type min = numeric_limits<Type>::min(),
        const Type max = numeric_limits<Type>::max()
    ) {
        if constexpr (is_integral_v<Type>) {
            return uniform_int_distribution<Type>(min, max)(mGenerator);
        } else if constexpr (is_floating_point_v<Type>) {
            return uniform_real_distribution<Type>(min, max)(mGenerator);
        } else {
            return {};
        }
    }

    template <typename Type>
    inline auto GetVector(
        const size_t size,
        const Type min = numeric_limits<Type>::min(),
        const Type max = numeric_limits<Type>::max()
    ) {
        return views::iota(0U, size) | views::transform([=] (auto _) { return Get(min, max); });
    }

    template <typename Type>
    inline auto GetMatrix(
        const size_t rows,
        const size_t columns = 1U,
        const Type min = numeric_limits<Type>::min(),
        const Type max = numeric_limits<Type>::max()
    ) {
        return views::iota(0U, rows) | views::transform([=] (auto _) { return GetVector(columns, min, max); });
    }

private:
    random_device mSeeder;
    mt19937 mGenerator;
};

class Backpack {
public:
    using backpack = struct backpack_s {
        vector<uint16_t> values {};
        vector<uint16_t> weights {};
        uint16_t weightMax = 0;
    };

    inline Backpack(const uint16_t chromosomeSize, const uint32_t populationSize, const uint16_t groupCount, const float pm)
        : mChromosomeSize(chromosomeSize), mPopulationSize(populationSize), mGroupCount(groupCount), mPM(pm) {}

    inline void SetBackpack(const vector<uint16_t>& values, const vector<uint16_t>& weights, const uint16_t weightMax) {
        mBackpack = { values, weights, weightMax };
    }

    void Simulate(const uint32_t generationCount) {
        cout << "Backpack values: " << endl;
        Print<uint16_t>(mBackpack.values);
        cout << "Backpack weights: " << endl;
        Print<uint16_t>(mBackpack.weights);
        cout << "Backpack max weight: " << endl;
        cout << mBackpack.weightMax << endl << endl;

        matrix<int> population;
        {
            auto populationView = move(mRandom.GetMatrix(mPopulationSize, mChromosomeSize, 0, 1));
            ranges::for_each(populationView, [&population] (const auto& rowView) { population.push_back(vector(rowView.begin(), rowView.end())); });
        }

        matrix<int> fitnessFirstAndLast {};

        for (size_t i = 0; i < generationCount; i++) {
            auto fitness = move(GenerateFitness(population));
            if (i == 0 || i == generationCount - 1) {
                fitnessFirstAndLast.push_back(fitness);
            }

            auto fitnessSegments = move(GenerateFitnessSegments(fitness));
            auto chromosomes = move(SelectChromosomes(population, fitnessSegments));
            auto populationNew = move(CrossPopulationAndMutateChildren(chromosomes));

            population = move(KillTheWEAKAndAddChildren(population, fitness, populationNew));

            cout << "Iteration " << i << ": medium = " << Mean(fitness) << ", dispersion = " << Dispersion(fitness) << endl;
        }

        cout << endl << "Fitness improvement percentage: " << (int)GetFitnessesImprovementPercentage(fitnessFirstAndLast) << '%' << endl;
    }

private:
    Random mRandom;

    uint16_t mChromosomeSize;
    uint32_t mPopulationSize;

    backpack mBackpack;

    uint16_t mGroupCount;
    const uint8_t mGroupSize = 2U;
    float mPM;

    vector<int> GenerateFitness(const matrix<int>& population) const {
        vector<int> fitness;

        for (const auto& chromosome : population) {
            auto chromosomeWeight = 0;
            auto chromosomeValue = 0;

            for (size_t i = 0; i < chromosome.size(); i++) {
                if (chromosome[i] == 1) {
                    chromosomeWeight += mBackpack.weights[i];
                    chromosomeValue += mBackpack.values[i];
                }
            }

            if (chromosomeWeight > mBackpack.weightMax) {
                chromosomeValue = 0;
            }

            fitness.push_back(chromosomeValue);
        }

        return fitness;
    }

    vector<float> GenerateFitnessSegments(const vector<int>& fitness) const {
        auto fitnessSum = accumulate(fitness.begin(), fitness.end(), 0.f);

        vector<float> probabilitiesOfSelection;
        ranges::for_each(fitness, [&] (const auto& item) { probabilitiesOfSelection.push_back(item / fitnessSum); });

        vector<float> fitnessSegments { 0.f };
        ranges::for_each(probabilitiesOfSelection, [&] (const auto& item) { fitnessSegments.push_back(fitnessSegments.back() + item); });

        return fitnessSegments;
    }

    vector<int> SelectChromosome(const matrix<int>& population, const vector<float>& fitnessSegments) {
        auto randFloat = mRandom.Get(0.f, fitnessSegments.back());

        for (size_t i = 0; i < fitnessSegments.size(); i++) {
            if (randFloat <= fitnessSegments[i]) {
                return population[i - 1];
            }
        }

        assert(false && "There is no segment that contains such a value!");
        return {};
    }

    matrix<int> SelectChromosomes(const matrix<int>& population, const vector<float>& fitnessSegments) {
        matrix<int> chromosomesGroups;

        for (size_t i = 0; i < (size_t)(mGroupCount * mGroupSize); i++) {
            chromosomesGroups.push_back(SelectChromosome(population, fitnessSegments));
        }

        return chromosomesGroups;
    }

    vector<int> MutateChild(vector<int> chromosomeChild) {
        auto pms = move(mRandom.GetVector(mChromosomeSize, 0.f, 1.f));

        for (size_t i = 0; i < ranges::distance(pms); i++) {
            if (pms[i] < mPM) {
                chromosomeChild[i] &= ~1;
            }
        }

        return chromosomeChild;
    }

    matrix<int> CrossChromosomes(const vector<int>& chromo1, const vector<int>& chromo2) {
        vector<int> chromosome1, chromosome2;

        for (size_t i = 0; i < chromo1.size(); i++) {
            if (mRandom.Get(0, 1)) {
                chromosome1.push_back(chromo1[i]);
                chromosome2.push_back(chromo2[i]);
            } else {
                chromosome2.push_back(chromo1[i]);
                chromosome1.push_back(chromo2[i]);
            }
        }

        return { chromosome1, chromosome2 };
    }

    matrix<int> CrossPopulationAndMutateChildren(const matrix<int>& chromoGroups) {
        matrix<int> chromoGroupsNew;

        for (size_t i = 0; i < chromoGroups.size(); i += mGroupSize) {
            auto children = move(CrossChromosomes(chromoGroups[i], chromoGroups[i + 1]));

            chromoGroupsNew.push_back(MutateChild(children[0]));
            chromoGroupsNew.push_back(MutateChild(children[1]));
        }

        return chromoGroupsNew;
    }

    inline float GetFitnessesImprovementPercentage(const matrix<int>& fitnesses) const {
        auto fitnessSumFirst = accumulate(fitnesses.front().begin(), fitnesses.front().end(), 0.f);
        auto fitnessSumLast = accumulate(fitnesses.back().begin(), fitnesses.back().end(), 0.f);

        return (fitnessSumLast - fitnessSumFirst) / fitnessSumLast * 100.f;
    }

    inline matrix<int> SortPopulationByFitness(const matrix<int>& population, const vector<int> fitnesses) {
        vector<tuple<vector<int>, int>> populationAndFitness;
        for (size_t i = 0; i < population.size(); i++) {
            populationAndFitness.push_back({ population[i], fitnesses[i] });
        }

        sort(populationAndFitness.begin(), populationAndFitness.end(), [] (const auto& left, const auto& right) { return get<1>(left) < get<1>(right); });

        matrix<int> populationSorted;
        ranges::for_each(populationAndFitness, [&] (const auto& item) { populationSorted.push_back(get<0>(item)); });

        return populationSorted;
    }

    inline matrix<int> KillTheWEAKAndAddChildren(const matrix<int>& population, const vector<int> fitnesses, const matrix<int>& populationNew) {
        auto populationSorted = move(SortPopulationByFitness(population, fitnesses));
        populationSorted.erase(populationSorted.begin(), populationSorted.begin() + mGroupCount * mGroupSize);
        populationSorted.insert(populationSorted.end(), populationNew.begin(), populationNew.end());

        return populationSorted;
    }

    inline float Mean(const vector<int>& fitness) const {
        return accumulate(fitness.begin(), fitness.end(), 0.f) / fitness.size();
    }

    inline float Dispersion(const vector<int>& fitness) const {
        auto mean = Mean(fitness);

        auto standardDeviation = 0.;
        ranges::for_each(fitness, [&] (const auto& elem) { standardDeviation += pow(elem - mean, 2); });
        return (float)sqrt(standardDeviation / fitness.size());
    }
};

int main() {
    Backpack backpack(10, 100, 18, 0.25);
    backpack.SetBackpack({ 30, 15, 60, 85, 100, 10, 25, 50, 5, 70 }, { 5, 14, 6, 8, 14, 11, 4, 11, 9, 20 }, 50);

    backpack.Simulate(100);

    return 0;
}
