#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <concepts>
#include <bit>

#include "debug.hpp"

inline static auto error = [](const std::string& msg) {
    perror(msg.c_str());
};

template<typename T>
requires std::integral<T> || std::floating_point<T>
T load_number(std::istream& is) {
    static_assert(std::endian::native == std::endian::little ||
                  std::endian::native == std::endian::big);
    // Load the little endian order binary representation
    T x;
    int size = sizeof(x);
    char* data = reinterpret_cast<char*>(&x);
    if constexpr (std::endian::native == std::endian::little) {
        for (int i = 0; i < size; i++) {
            is.read(&data[i], 1);
        }
    } else {
        for (int i = size - 1; i >= 0; i--) {
            is.read(&data[i], 1);
        }
    }
    return x;
}

template<typename T>
requires std::integral<T> || std::floating_point<T>
void save_number(std::ostream& os, T x) {
    static_assert(std::endian::native == std::endian::little ||
                  std::endian::native == std::endian::big);
    // Save the binary representation in little endian order
    int size = sizeof(x);
    char* data = reinterpret_cast<char*>(&x);
    if constexpr (std::endian::native == std::endian::little) {
        for (int i = 0; i < size; i++) {
            os << data[i];
        }
    } else {
        for (int i = size - 1; i >= 0; i--) {
            os << data[i];
        }
    }
}

inline static auto load = [](std::istream& is) {
    using namespace std;
    pair<vector<string>, vector<vector<float>>> res;
    int vocabulary_size = load_number<int>(is);
    dbg(vocabulary_size);
    for (int i = 0; i < vocabulary_size; i++) {
        string w;
        is >> w;
        res.first.emplace_back(w);
    }
    is.ignore();
    int embedding_size = load_number<int>(is);
    dbg(embedding_size);
    vector<vector<float>> syn0(vocabulary_size, vector<float>(embedding_size));
    float m = 0;
    for (int i = 0; i < vocabulary_size; i++) {
        for (int j = 0; j < embedding_size; j++) {
            syn0[i][j] = load_number<float>(is);
            m = std::max(m, abs(syn0[i][j]));
        }
    }
    dbg(m);
    res.second = std::move(syn0);
    return res;
};

template<typename T>
std::vector<T>& operator+=(std::vector<T>& v1, const std::vector<T>& v2) {
    if (v1.size() == 0) return v1 = v2;
    if (v2.size() == 0) return v1;
    assert(v1.size() == v2.size());
    for (int i = 0; i < (int)v1.size(); i++) {
        v1[i] += v2[i];
    }
    return v1;
}

template<typename T, typename U>
std::vector<T>& operator/=(std::vector<T>& v, U x) {
    for (int i = 0; i < (int)v.size(); i++) {
        v[i] /= x;
    }
    return v;
}

template<typename T, typename U>
std::vector<T> operator*(const std::vector<T>& v, U x) {
    std::vector<T> res = v;
    for (int i = 0; i < (int)res.size(); i++) {
        res[i] *= x;
    }
    return res;
}

template<typename T>
T sigmoid(T x) {
    return 1 / (1 + exp(-x));
}

template<typename T>
T sigmoid_derivative(T x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

template<typename T>
using weighted_vector = std::pair<std::vector<T>, int>;

template<typename T>
weighted_vector<T> make_weighted_vector(const std::vector<T> v) {
    return {v, 1};
}

template<typename T>
weighted_vector<T> normalize(const weighted_vector<T>& v) {
    if (v.second == 0) return v;
    weighted_vector<T> res = v;
    res.first /= res.second;
    res.second = 1;
    return res;
}

template<typename T>
weighted_vector<T>& operator+=(weighted_vector<T>& v1, const weighted_vector<T>& v2) {
    if (v1.second == 0) return v1 = v2;
    if (v2.second == 0) return v1;
    if (v1.first.size() != v2.first.size()) {
        dbg(v1, v2);
    }
    assert(v1.first.size() == v2.first.size());
    v1.first += v2.first;
    v1.second += v2.second;
    return v1;
}

template<typename T>
weighted_vector<T>& operator+=(weighted_vector<T>& v1, const std::vector<T>& v2) {
    if (v1.second == 0) {
        v1.first = v2;
        if (v2.size() == 0) {
            dbg(v2);
        }
        v1.second = 1;
        return v1;
    }
    assert(v1.first.size() == v2.size());
    v1.first += v2;
    v1.second++;
    return v1;
}

template<typename T>
std::vector<T>& operator+=(std::vector<T>& v1, const weighted_vector<T>& v2) {
    if (v2.second == 0) return v1;
    if (v1.size() != v2.first.size()) {
        dbg(v1, v2);
    }
    assert(v1.size() == v2.first.size());
    v1 += normalize(v2).first;
    return v1;
}

template<typename T, typename U>
weighted_vector<T>& operator/=(weighted_vector<T>& v, U x) {
    v.first /= x;
    return v;
}

template<typename T, typename U>
weighted_vector<T> operator*(const weighted_vector<T>& v, U x) {
    weighted_vector<T> res;
    res.first = v.first * x;
    res.second = v.second;
    return res;
}
