#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> maxPrimes(int n, int s) {
        vector<int> ret;
        if (n < 2 || s < 2) return ret;

        vector<bool> prime(n + 1, true);
        prime[0] = prime[1] = false;

        for (int i = 2; 1LL * i * i <= n; ++i) {
            if (!prime[i]) continue;
            for (long long j = 1LL * i * i; j <= n; j += i) {
                prime[j] = false;
            }
        }

        long long sum = 0;
        for (int i = 2; i <= n; ++i) {
            if (!prime[i]) continue;
            if (sum + i > s) break;

            ret.push_back(i);
            sum += i;
        }
        return ret;
    }
};
