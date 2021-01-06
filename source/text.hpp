#pragma once
#include <boost/program_options.hpp>
#include "debug.hpp"

namespace po = boost::program_options;

struct Text {
    inline static const std::set<std::string> stopwords {"unto", "le", "de", "la", "s", "still","should","very","for","quite","moreover","less","thereafter","thereupon","never","a","except","i","around","that","three","ourselves","as","had","over","six","almost","am","ours","others","latter","could","through","were","is","name","'ll","'re","where","then","least","can","call","us","last","was","behind","further","using","below","his","thence","your","whole","ca","did","wherein","give","yours","into","does","upon","nor","seeming","one","done","thus","hundred","not","empty","herself","four","yourselves","please","when","against","top","other","some","once","really","just","we","though","doing","own","off","our","onto","together","whether","he","since","else","even","see","the","beyond","serious","these","wherever","its","made","itself","has","mostly","seemed","alone","becoming","besides","side","beforehand","forty","neither","twenty","would","up","in","than","elsewhere","mine","sometime","front","regarding","yet","via","been","seems","my","therein","eight","nine","whatever","after","she","among","of","unless","who","such","beside","and","within","or","show","toward","any","all","either","ever","everywhere","if","while","sometimes","whenever","no","whereas","anyhow","hence","go","from","so","used","much","back","whose","although","five","everyone","re","whither","fifty","various","'m","by","anyone","many","whereby","with","those","why","always","few","will","another","rather","n't","during","here","fifteen","without","otherwise","anywhere","hereafter","nevertheless","out","whoever","be","hereby","also","again","thru","across","himself","both","noone","until","too","whereafter","along","myself","they","somewhere","therefore","none","per","on","afterwards","someone","their","are","nobody","move","towards","whom","enough","more","became","'s","you","sixty","them","becomes","about","hereupon","become","same","hers","meanwhile","due","being","amount","down","perhaps","have","yourself","themselves","which","to","well","namely","make","often","there","me","cannot","this","first","at","twelve","what","indeed","eleven","an","above","former","part","'d","put","full","nowhere","how","because","ten","latterly","third","under","before","get","next","seem","anyway","must","take","might","throughout","however","something","amongst","bottom","'ve","every","formerly","already","between","keep","may","somehow","two","whereupon","anything","say","several","but","do","each","him","herein","everything","it","most","only","thereby","whence","nothing","now","her",};
    std::vector<int> text;
    std::vector<std::string> vocabulary;
    std::vector<int> cnt;
    std::vector<int> unigram_table;
    std::vector<float> subsampling;

    Text(const po::variables_map& vm) {
        using namespace std;
        {
            // text, vocabulary, cnt
            vector<pair<string, int>> words;
            string filename = vm["train"].as<string>();
            ifstream is{filename, ios::binary};
            if (!is.is_open()) {
                error("error while opening text " + filename);
                return;
            }
            bool use_stop_words = vm.count("stop");
            for (int i = 0; !(is.eof() || is.bad());) {
                string w;
                is >> w;
                if (w == "" || (use_stop_words && stopwords.count(w))) continue;
                words.emplace_back(w, i++);
            }
            if (is.bad()) {
                error("error while reading text " + filename);
                return;
            }
            dbg(words.size());
            int min_count = vm["min-count"].as<int>();
            sort(execution::par_unseq, begin(words), end(words));
            dbg("sorted");
            vector<pair<int, int>> text;
            for (int i = 0; i < (int)words.size();) {
                std::string w = words[i].first;
                int k = upper_bound(begin(words) + i, end(words), make_pair(w, INT_MAX)) - begin(words);
                if (k - i >= min_count) {
                    this->vocabulary.emplace_back(w);
                    this->cnt.emplace_back(k - i);
                    int id = this->vocabulary.size() - 1;
                    for (int j = i; j < k; j++) {
                        text.emplace_back(words[j].second, id);
                    }
                }
                i = k;
            }
            dbg(this->vocabulary.size());
            sort(execution::par_unseq, begin(text), end(text));
            dbg(text.size());
            for (const auto& p : text) {
                this->text.emplace_back(p.second);
            }
            dbg(this->text.size());
#ifdef DEBUG
            for (int i = 0; i < 1000; i++) {
                cout << this->vocabulary[this->text[i]] << ' ';
            }
            cout << endl;
#endif
        }
        {
            // Unigram table
            constexpr float power = 0.75;
            constexpr int precision = 2e8;
            float words_power = accumulate(begin(this->cnt), end(this->cnt), 1,
                                           [&](float acc, int v) {
                                               return acc + pow(v, power);
                                           });
            int ix = 0;
            float p = pow(cnt[ix], power) / words_power;
            for (int i = 0;; i++) {
                this->unigram_table.emplace_back(ix);
                if (i > p * precision) {
                    ix++;
                    if (ix == (int)this->cnt.size()) break;
                    p += pow(this->cnt[ix], power) / words_power;
                }
            }
            dbg(unigram_table.size());
        }
        {
            // Subsampling
            float sample = vm["sample"].as<float>();
            if (sample == 0) return;
            this->subsampling.resize(this->vocabulary.size());
            for (int i = 0; i < (int)this->vocabulary.size(); i++) {
                float x = cnt[i] / (float)this->text.size();
                this->subsampling[i] = (sqrt(x / sample) + 1) * (sample / x);
                if (this->subsampling[i] < 1) dbg(this->subsampling[i]);
            }
        }
    }
};
