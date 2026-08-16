//
// Created by PLaYX on 2026/8/16.
//
#include <bits/stdc++.h>
using namespace std;
class Solution {
public:
    int nearestDrone(vector<vector<int>>& drones, vector<int>& target) {
        int ans = -1,ans_min = INT_MAX;
        int i = 0;
        for (auto drone : drones) {
            int dis = abs(drone[0] - target[0]) + abs(drone[1] - target[1]);
            if (dis <= drone[2] && dis < ans_min) {
                ans_min = dis;
                ans = i;
            }
            i++;
        }
        return ans;
    }
};