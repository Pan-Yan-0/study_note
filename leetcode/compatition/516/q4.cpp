//
// Created by PLaYX on 2026/8/23.
//
#include <bits/stdc++.h>
using namespace std;
class Solution {
public:
    vector<bool> validSubarrays(vector<int>& nums, int k, vector<vector<int>>& queries) {
        int n = static_cast<int>(nums.size());

        // prefix[i] 表示 nums[0..i-1] 中每个数字的频率
        vector<unordered_map<int, int>> prefix(n + 1);

        // 预处理所有前缀
        for (int i = 1; i <= n; ++i) {
            // 复制前一个前缀的完整频率表
            prefix[i] = prefix[i - 1];

            int value = nums[i - 1];
            ++prefix[i][value];
        }

        vector<bool> answer;
        answer.reserve(queries.size());

        for (const auto& query : queries) {
            int left = query[0];
            int right = query[1];

            int distinctCount = 0;
            bool allEven = true;

            // 遍历右前缀中出现过的所有数字
            for (const auto& [value, rightCount] : prefix[right + 1]) {
                int leftCount = 0;

                auto it = prefix[left].find(value);
                if (it != prefix[left].end()) {
                    leftCount = it->second;
                }

                // value 在 nums[left..right] 中的出现次数
                int rangeCount = rightCount - leftCount;

                if (rangeCount > 0) {
                    ++distinctCount;

                    if (rangeCount % 2 != 0) {
                        allEven = false;
                        break;
                    }
                }
            }

            answer.push_back(allEven && distinctCount == k);
        }

        return answer;
    }
};