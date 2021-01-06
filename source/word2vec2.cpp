#include <iostream>
#include <boost/program_options.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <concepts>
#include <ranges>
#include <algorithm>
#include <numeric>
#include <execution>
#include <random>
#include <memory>
#include <chrono>
#include "util.hpp"
#include "text.hpp"

namespace po = boost::program_options;

using gradient = std::array<std::vector<weighted_vector<float>>, 2>;

struct WordEmbedding {
    std::vector<std::vector<float>> syn0;
    std::vector<std::vector<float>> syn1neg1;
    std::shared_ptr<Text> text;

    WordEmbedding(int size, std::shared_ptr<Text> text) : text(text) {
        init(size, text->vocabulary.size());
    }

    void init(int size, int vocab_size) {
        using namespace std;
        syn0.resize(vocab_size, vector<float>(size));
        syn1neg1.resize(vocab_size, vector<float>(size));
        default_random_engine generator;
        uniform_real_distribution<float> distribution(-0.5, 0.5);
        for (auto& hidden2output : syn1neg1) {
            for (float& weight : hidden2output) {
                weight = distribution(generator);
            }
        }
    }

    int embedding_dim() const {
        return syn0[0].size();
    }

    gradient empty_gradient() const {
        std::vector<weighted_vector<float>> empty(text->vocabulary.size());
        return {empty, empty};
    }

    gradient learn(const std::pair<int, int>& slice, float starting_alpha,
                   int window, int negative, int iter, int max_iter) {
        using namespace std;
        const int vocab_size = text->vocabulary.size();
        vector<weighted_vector<float>> syn0_grad(vocab_size);
        vector<weighted_vector<float>> syn1neg1_grad(vocab_size);
        float alpha = starting_alpha * (1 - (float)iter / max_iter);
        default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
        uniform_real_distribution<float> distribution(0, 1);
        vector<int> sentence;
        // Subsampling of frequent words
        for (int i = slice.first; i < slice.second; i++) {
            if (text->subsampling.size() &&
                text->subsampling[text->text[i]] < distribution(generator)) continue;
            sentence.emplace_back(text->text[i]);
        }
        int sentence_length = sentence.size();
        for (int sentence_pos = 0; sentence_pos < sentence_length; sentence_pos++) {
            int word = sentence[sentence_pos];
            // This vector will hold the *average* of all of the context word vectors.
            // This is the output of the hidden layer.
            weighted_vector<float> neu1;
            // Holds the gradient for updating the hidden layer weights.
            // This same gradient update is applied to all context word vectors.
            weighted_vector<float> neu1e;
            for (int i = 0; i < 2 * window + 1; i++) {
                if (i == window) continue;
                int j = sentence_pos - window + i;
                if (j < 0) continue;
                if (j >= sentence_length) continue;
                int context_word = sentence[j];
                neu1 += syn0[context_word];
            }
            [[unlikely]] if (neu1.second == 0) continue;
            neu1 = normalize(neu1);
            for (int sample = 0; sample < negative + 1; sample++) {
                int target, label;
                if (sample == 0) {
                    target = word;
                    label = 1;
                } else {
                    uniform_int_distribution<int> negative_sample(0, (int)text->unigram_table.size() - 1);
                    do {
                        target = text->unigram_table[negative_sample(generator)];
                    } while (target == word);
                    label = 0;
                }
                float dot_product = inner_product(begin(neu1.first), end(neu1.first),
                                                  begin(syn1neg1[target]), static_cast<float>(0));
                float error = label - sigmoid(dot_product);
//                    error *= sigmoid_derivative(dot_product);
                neu1e += syn1neg1[target] * error;
                syn1neg1_grad[target] += neu1 * error * alpha;
            }
            for (int i = 0; i < 2 * window + 1; i++) {
                if (i == window) continue;
                int j = sentence_pos - window + i;
                if (j < 0) continue;
                if (j >= sentence_length) continue;
                int context_word = sentence[j];
                syn0_grad[context_word] += neu1e * alpha;
            }
        }
        return {std::move(syn0_grad), std::move(syn1neg1_grad)};
    }

    static gradient combine(const gradient& m1, const gradient& m2) {
        gradient res = m1;
        for (int k = 0; k < 2; k++) {
            for (int i = 0; i < res[k].size(); i++) {
                res[k][i] += m2[k][i];
            }
        }
        return res;
    }

    void update(gradient& m) {
        for (int i = 0; i < (int)syn0.size(); i++) {
            syn0[i] += normalize(m[0][i]);
        }
        for (int i = 0; i < (int)syn1neg1.size(); i++) {
            syn1neg1[i] += normalize(m[1][i]);
        }
    }

    void save(std::ostream& os) const {
        save_number(os, (int)text->vocabulary.size());
        for (const std::string& w : text->vocabulary) {
            os << w << ' ';
        }
        save_number(os, (int)syn0[0].size());
        int all_zeros = 0;
        float max_abs = 0;
        for (int i = 0; i < (int)syn0.size(); i++) {
            bool all = true;
            for (int j = 0; j < (int)syn0[0].size(); j++) {
                if (syn0[i][j] != 0) all = false;
                max_abs = std::max(max_abs, abs(syn0[i][j]));
                save_number(os, syn0[i][j]);
            }
            all_zeros += all;
        }
        dbg(max_abs);
        dbg(all_zeros);
    }

    void load(std::istream& is) {
        auto [vocabulary, embeddings] = ::load(is);
        int vocabulary_size = vocabulary.size();
        assert(vocabulary_size == text->vocabulary.size());
        dbg(vocabulary_size);
        int embedding_size = embeddings.size();
        dbg(embedding_size);
        init(embedding_size, vocabulary_size);
        syn0 = std::move(embeddings);
    }
};


int main(int ac, char* av[]) {
    using namespace std;
    po::options_description desc("CBOW Word Embedding");
    desc.add_options()
        ("help", "produce help message")
        ("train", po::value<std::string>(), "Input file to train the model")
        ("model", po::value<std::string>(), "Model file")
        ("output", po::value<std::string>(), "Output file to save the resulting word vectors")
        ("size", po::value<int>()->default_value(300), "Set size of word vectors; default is 300")
        ("window", po::value<int>()->default_value(5), "Set max skip length between words; default is 5")
        ("sample", po::value<float>()->default_value(1e-4), "Set threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled; default is 1e-4, useful range is (0, 1e-5)")
        ("negative", po::value<int>()->default_value(5), "Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)")
        ("min-count", po::value<int>()->default_value(10), "This will discard words that appear less than MIN-COUNT times; default is 10")
        ("iter", po::value<int>()->default_value(15), "Run more training iterations (default 15)")
        ("alpha", po::value<float>()->default_value(0.5), "Set the starting learning rate; default is 0.5")
        ("thread", po::value<int>()->default_value(60), "Number of threads, default is 60")
        ("work", po::value<int>()->default_value(40000), "Work load by thread, default is 40000")
        ("stop", "Filter out stop words from text");
    po::positional_options_description p;
    p.add("train", -1);
    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(desc).positional(p).run(), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << '\n';
        return EXIT_SUCCESS;
    }
    if (vm.count("train") == 0) {
        cout << "You must specify the name of the training file\n";
        return EXIT_FAILURE;
    }
    Text text{vm};
    vector<pair<int, int>> slices;
    int nb_threads = vm["thread"].as<int>();
    int slice = vm["work"].as<int>();
    int text_size = text.text.size();
    if (slice == -1) slice = (text_size + nb_threads - 1) / nb_threads;
    for (int i = 0; i < text_size; i += slice) {
        slices.emplace_back(i, min(text_size, i + slice));
    }
    int nb_slices = slices.size();
    dbg(slice);
//    dbg(slices);
    dbg(nb_slices);
    WordEmbedding res{vm["size"].as<int>(), make_shared<Text>(text)};
    if (vm.count("model")) {
        string filename = vm["model"].as<string>();
        ifstream is{filename, ios::binary};
        if (!is.is_open()) {
            error("error while opening model " + filename);
            cerr << "continuing with empty model\n";
        } else {
            res.load(is);
            if (is.bad()) {
                error("error while reading model " + filename);
                cerr << "continuing with empty model\n";
            }
        }
    }
    float alpha = vm["alpha"].as<float>();
    int window = max(1, vm["window"].as<int>());
    int negative = max(1, vm["negative"].as<int>());
    int iter = vm["iter"].as<int>();
    default_random_engine engine(chrono::system_clock::now().time_since_epoch().count());
    int nb_batches = (slices.size() + nb_threads - 1) / nb_threads;
    dbg(nb_batches);
    for (int i = 0; i < iter; i++) {
        shuffle(begin(slices), end(slices), engine);
        for (int j = 0; j < nb_slices; j += nb_threads) {
            auto batch_start = begin(slices) + j;
            auto batch_end = begin(slices) + min(j + nb_threads, nb_slices);
            gradient m = transform_reduce(execution::par_unseq, batch_start, batch_end,
                                          res.empty_gradient(),
                                          WordEmbedding::combine,
                                          [&](const pair<int, int>& slice) {
                                              return res.learn(slice, alpha, window, negative, i, iter);
                                          });
            res.update(m);
        }
        dbg(i);
    }
    if (vm.count("output")) {
        string filename = vm["output"].as<string>();
        ofstream os{filename, ios::binary};
        if (!os.is_open()) {
            error("error while opening file " + filename);
            return EXIT_FAILURE;
        }
        res.save(os);
        if (os.bad()) {
            error("error while saving the model in " + filename);
            return EXIT_FAILURE;
        }
    } else {
        res.save(cout);
    }
    return EXIT_SUCCESS;
}