#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <concepts>

template<typename T>
concept iterable = requires(T t) {
    begin(t);
    end(t);
};

template<typename T>
requires iterable<T> and (!std::same_as<T, std::string>)
std::ostream& operator<<(std::ostream& os, const T& v) {
    os << '{';
    std::string sep;
    for (const auto &x : v) {
        os << sep << x;
        sep = ',';
    }
    return os << '}';
}

template<typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& p) {
    return os << '(' << p.first << ", " << p.second << ')';
}

template<typename Head, typename... Tail>
void dbg_out(const Head& head, const Tail&...tail) {
    std::cerr << ' ' << head;
    if constexpr (sizeof...(tail) > 0) {
        dbg_out(tail...);
    } else {
        std::cerr << std::endl;
    }
}

#ifdef DEBUG
#define dbg(...) std::cerr << "(" << #__VA_ARGS__ << "):", dbg_out(__VA_ARGS__)
#else
#define dbg(...)
#endif
