//
// Created by PLaYX on 2026/8/23.
//
#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<int> > findDisappearedNumbers(vector<int> &nums, int lower, int upper) {
        unordered_map<int, int> mp;
        for (int num: nums) {
            mp[num]++;
        }
        vector<int> los;
        for (int i = lower; i <= upper; i++) {
            if (mp.find(i) == mp.end()) {
                los.push_back(i);
            }
        }
        vector<vector<int> > ret;
        if (los.size() == 0 ) return ret;
        int start = 0,cur = lower;
        for (int i = 1; i < los.size(); i++) {
            int l = los[i];
            if (l != cur + 1) {
                ret.push_back({start,cur});
                start = l;
            }
            cur = l;
        }
        ret.push_back({start,cur});
        return ret;
    }
};
