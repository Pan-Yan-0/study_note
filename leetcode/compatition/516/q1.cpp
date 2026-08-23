//
// Created by PLaYX on 2026/8/23.
//
#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool isPalindromic(string s) {
        string binary;
        binary.reserve(s.size() * 8);

        // 将每个字符转换成 8 位二进制
        for (char ch : s) {
            char c = ch;
            for (int bit = 7; bit >= 0; --bit) {
                binary.push_back(((c >> bit) & 1) ? '1' : '0');
            }
        }

        // 判断二进制字符串是否为回文串
        int left = 0;
        int right = binary.size() - 1;
        while (left < right) {
            if (binary[left] != binary[right]) {
                return false;
            }
            ++left;
            --right;
        }

        return true;
    }
};
