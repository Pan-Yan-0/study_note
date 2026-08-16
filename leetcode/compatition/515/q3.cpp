//
// Created by PLaYX on 2026/8/16.
//
#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int maximumGap(string skill, string station) {
        int n = station.size();
        vector<int> L(n), R(n);
        for (int i = 0, j = 0; i < n; i++) {
            if (skill[j] == station[i]) {
                L[j] = i;
                j++;
            }
        }
        for (int i = n - 1, j = skill.size() - 1; i >= 0; i--) {
            if (skill[j] == station[i]) {
                R[j] = i;
                j--;
            }
            if (j == -1) break;
        }
        int ans = 0;
        for (int i = 0; i < n - 1; i++) {
            ans = max(R[i + 1] - L[i],ans);
        }
        return ans;
    }
};

int main() {
    string station = "aaaa";
    string skill = "aa";
    Solution s;
    cout << s.maximumGap(station, skill) << endl;
    return 0;
}
