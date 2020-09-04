#ifndef ISOLATIONFOREST_H
#define ISOLATIONFOREST_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <iostream>
#include <random>
#include <vector>

namespace isolationforest {

    typedef struct TreeNode {
        std::shared_ptr<TreeNode> left;
        std::shared_ptr<TreeNode> right;

        int feature_index;
        int size;
        double feature_cutoff;
        bool external;

        TreeNode(int feature_index, int size, double feature_cutoff, bool external) : 
            feature_index(feature_index), 
            size(size), 
            feature_cutoff(feature_cutoff), 
            external(external) {};
        ~TreeNode() {};
    } TreeNode;

    typedef std::shared_ptr<TreeNode> TreeNodePtr;
    typedef std::vector<TreeNodePtr> TreeNodePtrVector;
    typedef struct Sample {
        std::vector<double> data;
        int index;
    } Sample;

    typedef std::shared_ptr<Sample> SamplePtr;
    typedef std::vector<SamplePtr> SamplePtrVector;

    std::tuple<double, double> minmax(const SamplePtrVector &samples, const std::vector<int> &indicies, int ft_index) {
        double mx = 0;
        double min = 0;
        int i=0;
        for (auto i : indicies) {
            double val = samples[i]->data[ft_index];

            if (i==0) {
                mx = val;
                min = val;
                continue;
            } else if (val > mx) {
                mx = val;
            } else if (val < min) {
                min = val;
            }
        }
        return  std::make_tuple(min, mx);
    }

    class RNG {
        private:
            struct xorshift128p_state { uint64_t a, b; };
            xorshift128p_state state;
        public:
            RNG() {
                std::random_device rd;
                std::mt19937 g(rd());
                std::uniform_int_distribution<> m_rand(0, 2147483647);
                state.a = m_rand(g);
                state.b = m_rand(g);
            };

            /* The state must be seeded so that it is not all zero */
            uint64_t xorshift128p()
            {
                uint64_t t = state.a;
                uint64_t const s = state.b;
                state.a = s;
                t ^= t << 23;       // a
                t ^= t >> 17;       // b
                t ^= s ^ (s >> 26); // c
                state.b = t;
                return t + s;
            };

            double randDouble(uint64_t x) {
                const union { uint64_t i; double d; } u = {.i = UINT64_C(0x3FF) << 52 | x >> 12 };
                return u.d - 1.0;
            }
    };

    class Forest {
        private:
            int max_depth;
            int feature_count;
            int n_estimators;
            double subsampling_rate;
            int tree_size;
            RNG rng;

            inline double H(int i) { return std::log(i) + 0.5772156649; };
            inline double c(int n) { return 2*H(n - 1) - (2*(n - 1)/n); };

            TreeNodePtr createTree(const SamplePtrVector &subsample, int depth, const std::vector<int> &indicies) {

                if ( (depth >= max_depth) || (indicies.size() <= 1) ) {
                    TreeNodePtr node = std::make_shared<TreeNode>(-1, indicies.size(), 0.0, true);
                    return node;
                }

                // randomly select feature
                int ft_index = rng.xorshift128p() % feature_count;

                // randomly select value within samples
                std::tuple<double, double> mnmx = minmax(subsample, indicies, ft_index);
                double cutoff = rng.randDouble(rng.xorshift128p()) * (std::get<1>(mnmx) - std::get<0>(mnmx)) + std::get<0>(mnmx);

                // create left & right samples
                std::vector<int> left = std::vector<int>();
                std::vector<int> right = std::vector<int>();
                
                for (auto i : indicies) {
                    if (subsample[i]->data[ft_index] >= cutoff)
                        right.push_back(i);
                    else
                        left.push_back(i);
                }

                // create new node, set properties and left & right trees
                TreeNodePtr node = std::make_shared<TreeNode>(ft_index, indicies.size(), cutoff, false);
                node->right = createTree(subsample, depth+1, right);
                node->left = createTree(subsample, depth+1, left);

                return node;
            };

            double pathLength(SamplePtr x, TreeNodePtr T, int e) {
                // external node
                if (T->external) {
                    if (T->size <= 1)
                        return e;
                    return e+c(T->size);
                }

                int feature_index = T->feature_index;
                double cutoff = T->feature_cutoff;

                if (x->data[feature_index] >= cutoff)
                    return pathLength(x, T->right, e+1);
                else
                    return pathLength(x, T->left, e+1);
            };

        public:
            TreeNodePtrVector forest;
            Forest(int md, int fc, int ne, double ssr) : 
                max_depth(md), 
                feature_count(fc), 
                n_estimators(ne),
                subsampling_rate(ssr) {
                  rng = RNG();
                };
            ~Forest() {};

            void createForest(const SamplePtrVector &samples) {
                int subsample_size = (int)samples.size()*subsampling_rate;
                tree_size = subsample_size;
                std::vector<int> indicies(subsample_size, 0);

                for (int est=0; est<n_estimators; ++est) {

                    // sample data randomly (resevoir sampling)
                    int i=0;
                    for (i=0; i<subsample_size; ++i)
                        indicies[i] = samples[i]->index;

                    double W = std::exp(std::log(rng.xorshift128p())/subsample_size);

                    while (i <= samples.size()) {
                        i = i + std::floor(std::log(rng.xorshift128p())/std::log(1-W)) + 1;
                        if (i <= samples.size()) {
                            int randint = (rng.xorshift128p() % subsample_size);
                            indicies[randint] = samples[i]->index;
                            W *= std::exp(std::log(rng.xorshift128p())/subsample_size);
                        }
                    }

                    // Create Tree & add to Forest
                    TreeNodePtr tree = createTree(samples, 0, indicies);
                    forest.push_back(std::move(tree));
                }
            };

            std::map<int, double> scoreSamples(const SamplePtrVector& samples) {
                std::map<int, double> scores = std::map<int, double>();

                for (auto s : samples) {
                    double Eh = 0.0;

                    for (auto t : forest)
                        Eh += pathLength(s,t,0);
                    Eh /= forest.size();

                    double score = std::pow(0.5, (Eh / c(tree_size)));

                    scores.insert({s->index, score});
                }
                return scores;
            };
    };

} // namespace isolationforest

#endif //#ifndef ISOLATIONFOREST_H