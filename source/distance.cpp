#include <iostream>
#include <boost/program_options.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <concepts>
#include <algorithm>
#include <numeric>
#include <execution>
#include <random>
#include <memory>
#include <bit>
#include "util.hpp"

namespace po = boost::program_options;

std::vector<float> normalize(const std::vector<float>& v) {
    float norm = std::sqrt(std::accumulate(begin(v), end(v), static_cast<float>(0),
                                                 [](float acc, float x) {
                                                     return acc + x * x;
                                                 }) + std::numeric_limits<float>::epsilon());
    std::vector<float> res = v;
    for (auto& x : res) x /= norm;
    return res;
}

float cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2) {
    auto nv1 = normalize(v1);
    auto nv2 = normalize(v2);
    return inner_product(begin(nv1), end(nv1), begin(nv2), static_cast<float>(0));
}

int main(int ac, char* av[]) {
    using namespace std;
    po::options_description desc("Distance with word embedding");
    desc.add_options()
        ("help", "produce help message")
        ("embeddings", po::value<std::string>(), "Input file containing the embeddings")
        ("neighbors", po::value<int>()->default_value(40), "Number of closest words that will be shown, default is 40");
    po::positional_options_description p;
    p.add("embeddings", -1);
    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(desc).positional(p).run(), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << '\n';
        return EXIT_SUCCESS;
    }
    if (vm.count("embeddings") == 0) {
        cout << "You must specify the name of the file containing the embeddings\n";
        return EXIT_FAILURE;
    }
    string filename = vm["embeddings"].as<string>();
    ifstream is{filename, ios::binary};
    const auto [vocabulary, embeddings] = load(is);
    dbg(embeddings[0]);
    map<string, int> word2index;
    for (int i = 0; i < (int)vocabulary.size(); i++) {
        word2index[vocabulary[i]] = i;
    }
    int k = vm["neighbors"].as<int>();
    while (true) {
        cout << "Enter word (EXIT to break): ";
        string w; cin >> std::ws >> w;
        if (w == "EXIT") break;
        if (!word2index.count(w)) {
            cout << "Out of dictionary word!\n";
            continue;
        }
        const auto& embedding = embeddings[word2index[w]];
        vector<pair<float, string>> neighbors;
        for (const auto& w : vocabulary) {
            neighbors.emplace_back(cosine_similarity(embedding, embeddings[word2index[w]]), w);
        }
        sort(execution::par_unseq, rbegin(neighbors), rend(neighbors));
        for (int i = 0; i < k; i++) {
            cout << neighbors[i].second << ' ' << neighbors[i].first << '\n';
        }
    }
    return EXIT_SUCCESS;
}