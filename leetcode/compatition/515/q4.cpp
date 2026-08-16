//
// Created by PLaYX on 2026/8/16.
//
#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long elevatorRequests(int n, int start, vector<vector<int>>& requests) {
        int m = requests.size();
        int full = (1 << m) - 1;
        const long long INF = (long long)4e18;

        // memo[mask][i]：dfs(mask, i) 的答案，-1 表示没算过
        // 第二维是"请求下标"（只有 16 个），千万别用楼层 n（可达 10⁹）
        vector<vector<long long>> memo(1 << m, vector<long long>(m, -1));

        function<long long(int, int)> dfs = [&](int mask, int i) -> long long {
            // 边界：mask 里只有 i 一个，说明 i 是第一个处理的，从 start 直接开过来
            if (mask == (1 << i))
                return max((long long)abs(requests[i][1] - start),
                           (long long)requests[i][0]);

            if (memo[mask][i] != -1) return memo[mask][i];   // 查缓存

            int prev = mask ^ (1 << i);          // 处理 i 之前的集合
            long long ret = INF;
            for (int j = 0; j < m; j++) {        // 枚举上一个处理的请求 j
                if (!(prev & (1 << j))) continue;
                ret = min(ret, max(dfs(prev, j) + (long long)abs(requests[j][1] - requests[i][1]),
                                   (long long)requests[i][0]));
            }
            return memo[mask][i] = ret;          // 写缓存
        };

        // 所有请求都处理完，最后一个可以是任何请求
        long long ans = INF;
        for (int i = 0; i < m; i++) ans = min(ans, dfs(full, i));
        return ans;
    }
};
