//
// Created by PLaYX on 2026/8/14.
//
#include <bits/stdc++.h>
using namespace std;

// class Solution {
// public:
//     int findKthLargest(vector<int> &nums, int k) {
//         priority_queue<int, vector<int>, greater<int> > pq;
//         for (int i = 0; i < nums.size(); i++) {
//             pq.push(nums[i]);
//         }
//         for (int i = 0; i < k-1; i++) {
//             pq.pop();
//         }
//         return pq.top();
//     }
// };

class Solution {
public:
    void quickSort(vector<int> &nums, int l, int r) {
        int n = nums.size();
        srand(time(NULL));
        int pivotIndex = rand() % (r - l + 1) + l;
        int pivot = nums[pivotIndex];
        // 将pivot移除到调整数组的范围之外。
        swap(nums[pivotIndex], nums[l]);
        // 2 3 1 5 6 0
        int i = l + 1, j = r;
        while (true) {
            while (i <= j && nums[i] < pivot) {
                i++;
            }
            while (i <= j && nums[j] > pivot) {
                j--;
            }
            if (i >= j) {
                break;
            }
            swap(nums[i], nums[j]);
            i++;
            j--;
        }
        swap(nums[j], nums[l]);
        return
    }

    int partition(vector<int> &nums, int l, int r) {
        int n = nums.size();
        srand(time(NULL));
        int pivotIndex = rand() % (r - l + 1) + l;
        int pivot = nums[pivotIndex];
        // 将pivot移除到调整数组的范围之外。
        swap(nums[pivotIndex], nums[l]);
        // 2 3 1 5 6 0
        int i = l + 1, j = r;
        while (true) {
            while (i <= j && nums[i] < pivot) {
                i++;
            }
            while (i <= j && nums[j] > pivot) {
                j--;
            }
            if (i >= j) {
                break;
            }
            swap(nums[i], nums[j]);
            i++;
            j--;
        }
        swap(nums[j], nums[l]);
        return j;
    }

    int findKthLargest(vector<int> &nums, int k) {
        srand(time(NULL));
        int n = nums.size();
        int k_last_index = n - k;
        int l = 0, r = n - 1;
        while (true) {
            int pivot_index = partition(nums, l, r);
            if (pivot_index == k_last_index) {
                return pivot_index;
            }else if (pivot_index > k_last_index) {
                r = pivot_index - 1;
            }else {
                l = pivot_index + 1;
            }
        }

    }
};
