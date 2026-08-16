//
// Created by PLaYX on 2026/8/14.
//
#include <bits/stdc++.h>
using namespace std;
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int,int> mp;
        for (int num : nums) mp[num]++;
        vector<pair<int,int>> v(mp.begin(),mp.end());
        ranges::sort(v,[](auto p1,auto p2) {
            return p1.second < p2.second;
        });
        vector<int> ret;
        for (int i = 0; i < k; ++i) {
            ret.push_back(v[i].first);
        }
        return ret;
    }
};