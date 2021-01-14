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
#include <thread>
#include <mutex>
#include "util.hpp"
#include "text.hpp"

namespace po = boost::program_options;

static std::thread::id print_id;

struct WordEmbedding {
    std::vector<std::mutex> syn0_mutex;
    std::vector<std::mutex> syn1neg1_mutex;
    std::vector<std::vector<float>> syn0;
    std::vector<std::vector<float>> syn1neg1;
    const Text& text;

    WordEmbedding(int size, const Text& text) : text(text) {
        init(size, text.vocabulary.size());
    }

    void init(int size, int vocab_size) {
        using namespace std;
        syn0_mutex = vector<mutex>(vocab_size);
        syn1neg1_mutex = vector<mutex>(vocab_size);
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

    void learn(const std::pair<int, int>& slice, float starting_alpha,
               int window, int negative, int max_iter) {
        using namespace std;
        const int vocab_size = text.vocabulary.size();
        default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
        uniform_real_distribution<float> distribution(0, 1);
        for (int iter = 0; iter < max_iter; iter++) {
            if (this_thread::get_id() == print_id) dbg(iter);
            vector<int> sentence;
            // Subsampling of frequent words
            for (int i = slice.first; i < slice.second; i++) {
                if (text.subsampling.size() &&
                    text.subsampling[text.text[i]] < distribution(generator)) continue;
                sentence.emplace_back(text.text[i]);
            }
            int sentence_length = sentence.size();
//            dbg(sentence_length);
            float alpha = starting_alpha * (1 - (float)iter / max_iter);
            for (int sentence_pos = 0; sentence_pos < sentence_length; sentence_pos++) {
                int word = sentence[sentence_pos];
                // This vector will hold the *average* of all of the context word vectors.
                // This is the output of the hidden layer.
                vector<float> neu1(embedding_dim());
                // Holds the gradient for updating the hidden layer weights.
                // This same gradient update is applied to all context word vectors.
                vector<float> neu1e(embedding_dim());
                int cw = 0;
                for (int i = 0; i < 2 * window + 1; i++) {
                    if (i == window) continue;
                    int j = sentence_pos - window + i;
                    if (j < 0) continue;
                    if (j >= sentence_length) continue;
                    int context_word = sentence[j];
                    syn0_mutex[context_word].lock();
                    neu1 += syn0[context_word];
                    syn0_mutex[context_word].unlock();
                    cw++;
                }
                [[unlikely]] if (cw == 0) continue;
                neu1 /= cw;
                for (int sample = 0; sample < negative + 1; sample++) {
                    int target, label;
                    if (sample == 0) {
                        target = word;
                        label = 1;
                    } else {
                        uniform_int_distribution<int> negative_sample(0, (int)text.unigram_table.size() - 1);
                        do {
                            target = text.unigram_table[negative_sample(generator)];
                        } while (target == word);
                        label = 0;
                    }
                    syn1neg1_mutex[target].lock();
                    float dot_product = inner_product(begin(neu1), end(neu1),
                                                      begin(syn1neg1[target]), (float)0);
                    float error = label - sigmoid(dot_product);
                    //error *= sigmoid_derivative(dot_product);
                    neu1e += syn1neg1[target] * error;
                    syn1neg1[target] += neu1 * error * alpha;
                    syn1neg1_mutex[target].unlock();
                }
                for (int i = 0; i < 2 * window + 1; i++) {
                    if (i == window) continue;
                    int j = sentence_pos - window + i;
                    if (j < 0) continue;
                    if (j >= sentence_length) continue;
                    int context_word = sentence[j];
                    syn0_mutex[context_word].lock();
                    syn0[context_word] += neu1e * alpha;
                    syn0_mutex[context_word].unlock();
                }
            }
        }
    }

    void save(std::ostream& os) const {
        save_number(os, (int)text.vocabulary.size());
        for (const std::string& w : text.vocabulary) {
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
            if (all) dbg(text.vocabulary[i]);
            all_zeros += all;
        }
        dbg(max_abs);
        dbg(all_zeros);
    }

    void load(std::istream& is) {
        auto [vocabulary, embeddings] = ::load(is);
        int vocabulary_size = vocabulary.size();
        assert(vocabulary_size == text.vocabulary.size());
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
        ("alpha", po::value<float>()->default_value(0.001), "Set the starting learning rate; default is 0.001")
        ("thread", po::value<int>()->default_value(12), "Number of threads, default is 12")
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
    int text_size = text.text.size();
    int slice = (text_size + nb_threads - 1) / nb_threads;
    for (int i = 0; i < nb_threads; i++) {
        slices.emplace_back(i * slice, min(text_size, (i + 1) * slice));
    }
    dbg(slices);
    dbg(slice);
    WordEmbedding res{vm["size"].as<int>(), text};
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
    vector<thread> workers;
    for (int i = 0; i < nb_threads; i++) {
        workers.emplace_back([&, i] { res.learn(slices[i], alpha, window, negative, iter); });
    }
    print_id = workers[0].get_id();
    for (auto& worker : workers) {
        worker.join();
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