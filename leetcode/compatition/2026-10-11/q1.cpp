#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool threeFibonacciSum(int n) {
        if(n % 2 == 1) return false;
        if(n == 2 || n == 4) return true;
        int n1 = 2,n2 = 4,n3 = 0;
        while(n3 < n) {
            n3 = n1+n2;
            n1 = n2;
            n2 = n3;
            if(n3 == n) return true;
        }
        return false;
    }
};
