#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> maxPrimeSubset(int n, int s) {
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
        vector<vector<int>> dp(n+2,vector<int>(s+1,1001)); // dp[i][t]代表从i（包括i）之后可以达到t的最少质数数
        dp[n+1][0] = 0;

        // 选与不选状态转移：dp[i][t] = min(dp[i+1][t],(prime[i] == true启动分支)dp[i+1][t-i] + 1);
        // 一般来说状态转移依靠递归得出,故dp从i的max开始也就是递归叶节点
        for(int i = n;i >= 0;i--) {
            for(int t = 0;t <= s;t++) {
                dp[i][t] = dp[i+1][t];
                if(prime[i] && t - i >=0) dp[i][t] = min(dp[i][t],dp[i+1][t-i] + 1);
            }
        }
        // search target
        int target = -1;
        for(int i = s;i >= 0 && target == -1;i--) {
            if(dp[0][i] < 1001) {
                target = i;
            }
        }
        if(target == 0) return ret;
        // rebuild
        for(int i = 1;i <= n;i++) {
            if(prime[i] && target >= i && dp[i][target] == 1 + dp[i+1][target - i]) {
                ret.push_back(i);
                target -=i;
            }
        }
        return ret;
    }
};
