//
// Created by PLaYX on 2026/8/23.
//
#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int longestSubarray(vector<int>& nums, int k) {
        vector<int> primes = getPrimes(1000);

        vector<vector<int>> factors(nums.size());
        int maxValue = *max_element(nums.begin(), nums.end());

        for (int i = 0; i < static_cast<int>(nums.size()); ++i) {
            factors[i] = getPrimeFactors(nums[i], primes);
        }


        vector<int> mp(maxValue + 1, 0);

        int left = 0;
        int distinctPrimes = 0;
        int answer = 0;

        for (int right = 0; right < static_cast<int>(nums.size()); ++right) {

            for (int p : factors[right]) {
                if (mp[p] == 0) {
                    ++distinctPrimes;
                }
                ++mp[p];
            }

            while (distinctPrimes > k) {
                for (int p : factors[left]) {
                    --mp[p];
                    if (mp[p] == 0) {
                        --distinctPrimes;
                    }
                }
                ++left;
            }

            answer = max(answer, right - left + 1);
        }

        return answer;
    }

private:

    vector<int> getPrimes(int limit) {
        vector<bool> isComposite(limit + 1, false);
        vector<int> primes;

        for (int i = 2; i <= limit; ++i) {
            if (!isComposite[i]) {
                primes.push_back(i);

                for (long long j = 1LL * i * i; j <= limit; j += i) {
                    isComposite[j] = true;
                }
            }
        }

        return primes;
    }

    vector<int> getPrimeFactors(int x, const vector<int>& primes) {
        vector<int> result;

        for (int p : primes) {
            if (1LL * p * p > x) {
                break;
            }

            if (x % p == 0) {
                result.push_back(p);
                while (x % p == 0) {
                    x /= p;
                }
            }
        }
        if (x > 1) {
            result.push_back(x);
        }

        return result;
    }
};