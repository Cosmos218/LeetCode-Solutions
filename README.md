# LeetCode #

<!-- TOC -->

- [LeetCode](#leetcode)
    - [1.Two Sum](#1two-sum)
    - [3. Longest Substring Without Repeating Characters](#3-longest-substring-without-repeating-characters)
    - [11. Container With Most Water](#11-container-with-most-water)
    - [14. Longest Common Prefix](#14-longest-common-prefix)
    - [15. 3Sum](#15-3sum)
    - [16. 3Sum Closest](#16-3sum-closest)
    - [26. Remove Duplicates from Sorted Array](#26-remove-duplicates-from-sorted-array)
    - [49. Group Anagrams](#49-group-anagrams)
    - [53. Maximum Subarray](#53-maximum-subarray)
    - [66. Plus One](#66-plus-one)
    - [69. Sqrt(x)](#69-sqrtx)
    - [70. Climbing Stairs](#70-climbing-stairs)
    - [73. Set Matrix Zeroes](#73-set-matrix-zeroes)
    - [88. Merge Sorted Array](#88-merge-sorted-array)
    - [98. Validate Binary Search Tree](#98-validate-binary-search-tree)
    - [104. Maximum Depth of Binary Tree](#104-maximum-depth-of-binary-tree)
    - [118. Pascal's Triangle](#118-pascals-triangle)
    - [119. Pascal's Triangle II](#119-pascals-triangle-ii)
    - [121. Best Time to Buy and Sell Stock](#121-best-time-to-buy-and-sell-stock)
    - [122. Best Time to Buy and Sell Stock II](#122-best-time-to-buy-and-sell-stock-ii)
    - [136. Single Number](#136-single-number)
    - [167. Two Sum II - Input array is sorted](#167-two-sum-ii---input-array-is-sorted)
    - [169. Majority Element](#169-majority-element)
    - [198. House Robber](#198-house-robber)
    - [219. Contains Duplicate II](#219-contains-duplicate-ii)
    - [268. Missing Number](#268-missing-number)
    - [278. First Bad Version(binary search)](#278-first-bad-versionbinary-search)
    - [283. Move Zeroes](#283-move-zeroes)
    - [349. Intersection of Two Arrays](#349-intersection-of-two-arrays)
    - [350. Intersection of Two Arrays II](#350-intersection-of-two-arrays-ii)
    - [414. Third Maximum Number](#414-third-maximum-number)
    - [448. Find All Numbers Disappeared in an Array](#448-find-all-numbers-disappeared-in-an-array)
    - [485. Max Consecutive Ones](#485-max-consecutive-ones)
    - [532. K-diff Pairs in an Array](#532-k-diff-pairs-in-an-array)

<!-- /TOC -->

## 1.Two Sum ##

``` java
/********************************************************************************** 
* 
* Given an array of integers, find two numbers such that they add up to a specific target number.
* 
* The function twoSum should return indices of the two numbers such that they add up to the target, 
* where index1 must be less than index2. Please note that your returned answers (both index1 and index2) 
* are not zero-based.
* 
* You may assume that each input would have exactly one solution.
* 
* Input: numbers={2, 7, 11, 15}, target=9
* Output: index1=1, index2=2
* 
*               
**********************************************************************************/

```

``` python
class Solution:
    def twoSum(self, nums: 'List[int]', target: 'int') -> 'List[int]':
        dic = {}
        for i in range(len(nums)):
            if nums[i] in dic:
                return [dic[nums[i]], i]
            else:
                dic[target - nums[i]] = i
        return []
```

核心在于遍历的时候记录与当前数相加可以得到target的数，并放于字典中。如果后面的数有，就能立刻找到并返回index。

## 3. Longest Substring Without Repeating Characters ##

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        i, j, maxlength = 0, 0, 0
        subset = set()
        while i < len(s) and j < len(s):
            if s[j] not in subset:
                subset.add(s[j])
                j += 1
                maxlength = max(maxlength, j-i)
            else:
                subset.remove(s[i])
                i += 1
        return maxlength
```

核心要点：

1. 双指针指代substring范围
2. 双指针内的substring维持一个set，每次check右指针所指元素是否存在于set中，若存在则减小左指针，否则增加右指针。
3. 左右指针包含的范围被称为sliding window, 我们需要记录该window的最大范围作为返回值

## 11. Container With Most Water ##

Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

Note: You may not slant the container and n is at least 2.

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        if len(height) <= 1:
            return 0
        left, right = 0, len(height) - 1
        maxArea = float('-Inf')
        while left < right:
            current = min(height[left], height[right]) * (right - left)
            if current > maxArea:
                maxArea = current
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return maxArea
```

The key idea is that when fixing two pointers, the only way to increase the area is to move the shorter one inside because the area is determined by shorter one and when moving towards inside the length will decrease.

This problem can be considered by two pointers and greedy approach.

## 14. Longest Common Prefix ##

```python
# Write a function to find the longest common prefix string amongst an array of strings.

# If there is no common prefix, return an empty string "".

# Example 1:

# Input: ["flower","flow","flight"]
# Output: "fl"

# Solve it using vertical scanning
class Solution:
    def longestCommonPrefix(self, strs: 'List[str]') -> 'str':
        if not strs or len(strs) == 0:
            return ''
        for i in range(len(strs[0])):
            c = strs[0][i]
            for j in range(1, len(strs)):
                if len(strs[j]) <= i or strs[j][i] != c:
                    return strs[0][:i]
        return strs[0]
```

Follow up question:
Given a set of keys $S = [S_1,S_2 \ldots S_n]$, find the longest common prefix among a string q and S. This LCP query will be called frequently.

Since the LCP will be frequently called, we need to construct a Trie to store the set. **Need to be done**

## 15. 3Sum ##

```python
# Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

# Note:

# The solution set must not contain duplicate triplets.
class Solution:
    def threeSum(self, nums: 'List[int]') -> 'List[List[int]]':
        nums.sort()
        result = []
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            sum = 0 - nums[i]
            lo, hi = i + 1, len(nums) - 1
            while lo < hi:
                if nums[lo] + nums[hi] == sum:
                    result.append((nums[i], nums[lo], nums[hi]))
                    while lo < hi and nums[lo + 1] == nums[lo]:
                        lo += 1
                    while lo < hi and nums[hi - 1] == nums[hi]:
                        hi -= 1
                    lo += 1
                    hi -= 1
                elif nums[lo] + nums[hi] < sum:
                    lo += 1
                else:
                    hi -= 1
        return result
```

Key idea is first select one element, and choose two elements from the rest list to do two sums. Have to avoid duplicate when iterate both the first one and the second one.

## 16. 3Sum Closest ##

Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. Return the sum of the three integers. You may assume that each input would have exactly one solution.

Example:

Given array nums = [-1, 2, 1, -4], and target = 1.

The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

```java
class Solution {
    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int minSum = 0;
        int minDiff = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length - 2; i++) {
            int sum = target - nums[i];
            int lo = i + 1, hi = nums.length - 1;
            while (lo < hi) {
                int temp = nums[lo] + nums[hi];
                if (temp < sum) {
                    if (sum - temp < minDiff) {
                        minDiff = sum - temp;
                        minSum = temp + nums[i];
                    }
                    while (lo < hi && nums[lo] == nums[lo+1]) lo++;
                    lo++;
                } else if (temp > sum) {
                    if (temp - sum < minDiff) {
                        minDiff = temp - sum;
                        minSum = temp + nums[i];
                    }
                    while (lo < hi && nums[hi] == nums[hi-1]) hi--;
                    hi--;
                } else {
                    return target;
                }
            }
        }
        return minSum;
    }
}
```

Similar to 3Sum, the key idea is to firstly sort, secondly fix a num and find other two in the remaining sorted array using two pointers.

## 26. Remove Duplicates from Sorted Array ##

``` java
/********************************************************************************** 
* 
* Given a sorted array, remove the duplicates in place such that each element appear 
* only once and return the new length.
* 
* Do not allocate extra space for another array, you must do this in place with constant memory.
* 
* For example,
* Given input array A = [1,1,2],
* 
* Your function should return length = 2, and A is now [1,2].
* 
**********************************************************************************/
```

``` python
class Solution:
    def removeDuplicates(self, nums: 'List[int]') -> 'int':
        if len(nums) == 0:
            return 0
        i = 1
        for j in range(1,len(nums)):
            if nums[j-1] != nums[j]:
                nums[i] = nums[j]
                i += 1
        return i
```

核心： 双指针，一个遍历list，一个指向赋值的点

## 49. Group Anagrams ##

```python
# Given an array of strings, group anagrams together.

# Example:

# Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
# Output:
# [
#   ["ate","eat","tea"],
#   ["nat","tan"],
#   ["bat"]
# ]
class Solution:
    def groupAnagrams(self, strs: 'List[str]') -> 'List[List[str]]':
        dic = {}
        result = []
        for str in strs:
            sortedstr = ' '.join(sorted(str))
            if sortedstr in dic:
                result[dic[sortedstr]].append(str)
            else:
                temp = []
                temp.append(str)
                result.append(temp)
                dic[sortedstr] = len(result)-1
        return result
```

以上是我的首次AC方法，大体思路正确，小细节可以优化。
优化后的版本：

```python
class Solution(object):
    def groupAnagrams(self, strs):
        ans = collections.defaultdict(list)
        for s in strs:
            ans[tuple(sorted(s))].append(s)
        return ans.values()
```

其中collections.defaultdict(list)可以将dict的value类型设为list，因此即使为空也能append。 tuple(sorted(s))是为了将sorted的list类型的字符串转换为不可变类型tuple，which is hashable。

## 53. Maximum Subarray ##

Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Example:

Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.

```java
class Solution {
    public int maxSubArray(int[] nums) {
        // optHelper[i] = Math.max(nums[i], optHelper[i-1] + nums[i]);
        // opt[i] = Math.max(opt[i-1], optHelper[i]);
        int optHelper = 0, opt = Integer.MIN_VALUE;
        for (int i = 0; i < nums.length; i++) {
            optHelper = Math.max(nums[i], optHelper + nums[i]);
            opt = Math.max(opt, optHelper);
        }
        return opt;
    }
}
```

## 66. Plus One ##

```python
# Given a non-empty array of digits representing a non-negative integer, plus one to the integer.

# The digits are stored such that the most significant digit is at the head of the list, and each element in the array contain a single digit.

# You may assume the integer does not contain any leading zero, except the number 0 itself.

class Solution:
    def plusOne(self, digits: 'List[int]') -> 'List[int]':
        carry = 0
        for i in range(len(digits)-1, -1, -1):
            if digits[i] < 9:
                digits[i] += 1
                return digits
            else:
                digits[i] = 0
        if digits[0] == 0:
            digits.insert(0, 1)
        return digits
```

基本数组操作

## 69. Sqrt(x) ##

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        lo, hi = 0, x
        while lo < hi:
            mid = lo + (hi - lo + 1) // 2
            if mid * mid <= x:
                lo = mid
            else:
                hi = mid - 1
        return int(lo)
```

binary search to find the rightmost
为什么不能用另一种方法？

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        lo, hi = 0, x
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if mid * mid <= x:
                lo = mid + 1
            else:
                hi = mid
        return int(lo) - 1
```

因为这个无法处理x = 0 的情况，即初始时lo = hi。

## 70. Climbing Stairs ##

You are climbing a stair case. It takes n steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Note: Given n will be a positive integer.

```java
class Solution {
    public int climbStairs(int n) {
        if (n == 1) return 1;
        if (n == 2) return 2;
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
}
```

## 73. Set Matrix Zeroes ##

```python
# Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in-place.
class Solution:
    def setZeroes(self, matrix: 'List[List[int]]') -> 'None':
        """
        Do not return anything, modify matrix in-place instead.
        """
        col0 = 1
        row, col = len(matrix), len(matrix[0])
        for i in range(row):
            if matrix[i][0] == 0:
                col0 = 0
            for j in range(1, col):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
                
        for i in range(row - 1, -1, -1):
            for j in range(col - 1, 0, -1):
                if matrix[i][0] == 0 or matrix[0][j] == 0 :
                    matrix[i][j] = 0
            if col0 == 0:
                matrix[i][0] = 0
```

用数组的第一行和第一列来标记该行和该列是否为0.需要一个额外的变量来记录第一列，否则第一列和第一行都用matrix[0][0]。 第一遍遍历所有元素，只把第一列的记录在额外变量中，其他都在第一行和第一列。第二遍遍历，bottom-up, 第一列的元素根据额外变量置0，其他元素根据行首列首元素置零。

## 88. Merge Sorted Array ##

```python
# Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

# Note:

# The number of elements initialized in nums1 and nums2 are m and n respectively.
# You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2.

class Solution:
    def merge(self, nums1: 'List[int]', m: 'int', nums2: 'List[int]', n: 'int') -> 'None':
        """
        Do not return anything, modify nums1 in-place instead.
        """
        i, j = m-1, n-1
        k = m + n - 1
        while (i >= 0 and j >= 0):
            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]  
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1
        while (j >= 0):
            nums1[k] = nums2[j]
            j -= 1
            k -= 1
```

核心思想是从后往前放元素，这样就能保证num1的元素不会被覆盖掉。注意最后一个while只需要考虑nums2因为nums1的元素本来就在那里。

## 98. Validate Binary Search Tree ##

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        stack, previous = [], float('-Inf')
        
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if root.val <= previous:
                return False
            previous = root.val
            root = root.right
        
        return True
```

法1：前序遍历输出必须是从大到小。法2：递归调用，维持一个上界和下界比较

## 104. Maximum Depth of Binary Tree ##

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        return max(self.maxDepth(root.left) + 1, self.maxDepth(root.right) + 1)

```

## 118. Pascal's Triangle ##

Given a non-negative integer numRows, generate the first numRows of Pascal's triangle.

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        result = [[1]*(i+1) for i in range(numRows)]
        for i in range(1, numRows): # no need to care about first one otherwise index overflow
            for j in range(1, i, 1): # only care about middle ones
                result[i][j] = result[i-1][j-1] + result[i-1][j]
            # result.append(temp)
        return result
```

1. Python can init list with for loop inside, need to use this pythonic charac effieciently
2. When using Python, be careful about the boundary of the range: [)

## 119. Pascal's Triangle II ##

Given a non-negative index k where k ≤ 33, return the kth index row of the Pascal's triangle.

Note that the row index starts from 0.

```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        pascal = [0]*(rowIndex + 1)
        pascal[0] = 1
        for i in range(1, rowIndex + 1):
            for j in range(i, 0, -1):
                pascal[j] += pascal[j-1]
        return pascal
```

Need to care about the boundary cases when using python. 

## 121. Best Time to Buy and Sell Stock ##

Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Note that you cannot sell a stock before you buy one.

Example 1:

Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        maxProfit, minPrice = 0, float('+Inf')
        for price in prices:
            if price < minPrice:
                minPrice = price
            if price - minPrice > maxProfit:
                maxProfit = price - minPrice
        return maxProfit
```

A DP approach:

```java
class Solution {
    public int maxProfit(int[] prices) {
        int min = Integer.MAX_VALUE;
        int max = 0;
        for (int i = 0; i < prices.length; i++) {
            max = Math.max(max, prices[i] - min);
            min = prices[i] < min ? prices[i] : min;
        }
        return max;
    }
}
```

## 122. Best Time to Buy and Sell Stock II ##

``` java
/********************************************************************************** 
* 
* Say you have an array for which the ith element is the price of a given stock on day i.
* 
* Design an algorithm to find the maximum profit. You may complete as many transactions 
* as you like (ie, buy one and sell one share of the stock multiple times). However, 
* you may not engage in multiple transactions at the same time (ie, you must sell the 
* stock before you buy again).
*               
**********************************************************************************/
```

``` python
class Solution:
    def maxProfit(self, prices: 'List[int]') -> 'int':
        i = 0
        maxPrice = 0
        for j in range(1, len(prices)):
            if prices[j] > prices[j-1]:
                maxPrice += prices[j] - prices[j-1]
        return maxPrice
```

前后差值为正的时候加上即可

## 136. Single Number ##

```java
/********************************************************************************** 
* 
* Given an array of integers, every element appears twice except for one. Find that single one.
* 
* Note:
* Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
* 
*               
**********************************************************************************/
class Solution {
    public int singleNumber(int[] nums) {
        int result = 0;
        for (int i = 0; i < nums.length; i++) {
            result ^= nums[i];
        }
        return result;
    }
}
```

位运算骚操作之异或消掉两个相同位

## 167. Two Sum II - Input array is sorted ##

Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.

The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2.

Note:

Your returned answers (both index1 and index2) are not zero-based.
You may assume that each input would have exactly one solution and you may not use the same element twice.

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        dic = {}
        for i in range(len(numbers)):
            if numbers[i] not in dic:
                dic[target - numbers[i]] = i
            else:
                return [dic[numbers[i]] + 1, i + 1]
```

A improved python-HashMap solution, using enumerate:

```python
# dictionary
def twoSum2(self, numbers, target):
    dic = {}
    for i, num in enumerate(numbers):
        if target-num in dic:
            return [dic[target-num]+1, i+1]
        dic[num] = i
```

Other solution using two pointers and binary search:

```python 
# two-pointer
def twoSum1(self, numbers, target):
    l, r = 0, len(numbers)-1
    while l < r:
        s = numbers[l] + numbers[r]
        if s == target:
            return [l+1, r+1]
        elif s < target:
            l += 1
        else:
            r -= 1


# binary search        
def twoSum(self, numbers, target):
    for i in xrange(len(numbers)):
        l, r = i+1, len(numbers)-1
        tmp = target - numbers[i]
        while l <= r:
            mid = l + (r-l)//2
            if numbers[mid] == tmp:
                return [i+1, mid+1]
            elif numbers[mid] < tmp:
                l = mid+1
            else:
                r = mid-1
```

## 169. Majority Element ##

Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        dic = {}
        n = len(nums)
        for num in nums:
            dic[num] = dic.get(num, 0) + 1
            if dic[num] > n // 2:
                return num
```

**Note**: 记住python dic.get(num, 0)这个操作，是如果有key就取出，没有就置零并返回0

## 198. House Robber ##

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

Example 1:

Input: [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
             Total amount you can rob = 1 + 3 = 4.

```java
// A iterative memo solution
class Solution {
    public int rob(int[] nums) {
        // dp[i] = max(dp[i-2] + nums[i], dp[i-1])
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        if (nums.length == 2) return Math.max(nums[0], nums[1]);
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i-2] + nums[i], dp[i-1]);
        }
        return dp[nums.length-1];
    }
}
```

```java
// A iterative + 2 variables solution
class Solution {
    public int rob(int[] nums) {
        // dp[i] = max(dp[i-2] + nums[i], dp[i-1])
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        if (nums.length == 2) return Math.max(nums[0], nums[1]);
        int dp1 = nums[0];
        int dp2 = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            int temp = dp2;
            dp2 = Math.max(dp1 + nums[i], dp2);
            dp1 = temp;
        }
        return dp2;
    }
}
```

## 219. Contains Duplicate II ##

Given an array of integers and an integer k, find out whether there are two distinct indices i and j in the array such that nums[i] = nums[j] and the absolute difference between i and j is at most k.

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        dic = {}
        for i, num in enumerate(nums):
            if num not in dic:
                dic[num] = i
            else:
                if i - dic[num] <= k:
                    return True
                else:
                    dic[num] = i
        return False
```

利用enumerate提高代码简洁性

## 268. Missing Number ##

Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        result = len(nums)
        for i, num in enumerate(nums):
            result ^= i ^ num
        return result
```

## 278. First Bad Version(binary search) ##

```java
/**********************************************************************************
 *
 * The code base version is an integer start from 1 to n.
 * One day, someone committed a bad version in the code case, so it caused this version and the following versions are all failed in the unit tests.
 * Find the first bad version.
 *
 * You can call isBadVersion to help you determine which version is the first bad one.
 * The details interface can be found in the code's annotation part.
 *
 **********************************************************************************/
 /* The isBadVersion API is defined in the parent class VersionControl.
      boolean isBadVersion(int version); */

public class Solution extends VersionControl {
    public int firstBadVersion(int n) {
        int lo = 1, hi = n;
        while (lo < hi) {
            int mid = lo + (hi - lo)/2; //In order to avoid (hi + lo) overflow int, we use this
            if (isBadVersion(mid)) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }
}
```

首先，如果是找firstBadversion， 那么当mid是bad时，应该将hi减小至mid，这样才能不断逼近first的临近点。
其次，最应该考虑到的边界情况是，mid为first bad的时候。在此段代码中，因为是向下取整，所以hi取到firstbad后再算出来的mid都取不到该firstbad，最终由lo = mid + 1 = lo + 1 = hi结束，返回正确值。

如果是lastbadversion呢？有两种解决方法：

1. 将mid取整改为向上取整，同时让lo取mid：

```java
public class Solution extends VersionControl {
    public int lastBadVersion(int n) {
        int lo = 1, hi = n;
        while (lo < hi) {
            int mid = lo + (hi - lo + 1)/2; //In order to avoid (hi + lo) overflow int, we use this
            if (isBadVersion(mid)) lo = mid;
            else hi = mid - 1;
        }
        return lo;
    }
}
```

2. 做一些微小的操作（感觉不靠谱）:

```java
public class Solution extends VersionControl {
    public int lastBadVersion(int n) {
        int lo = 1, hi = n;
        while (lo < hi) {
            int mid = lo + (hi - lo)/2; //In order to avoid (hi + lo) overflow int, we use this
            if (isBadVersion(mid)) lo = mid + 1;
            else hi = mid;
        }
        return lo - 1;
    }
}
```

## 283. Move Zeroes ##

```python
/*************************************************************************************** 
 *
 * Given an array nums, write a function to move all 0's to the end of it while 
 * maintaining the relative order of the non-zero elements.
 *
 * For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should
 * be [1, 3, 12, 0, 0].
 * 
 * Note:
 * You must do this in-place without making a copy of the array.
 * Minimize the total number of operations.
 * 
 * Credits:
 * Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.
 *               
 ***************************************************************************************/

# my first version
 class Solution:
    def moveZeroes(self, nums: 'List[int]') -> 'None':
        """
        Do not return anything, modify nums in-place instead.
        """
        i, j = 0, 0
        while j < len(nums):
            while (j < len(nums) and nums[j] == 0):
                j += 1
            if j == len(nums):
                break
            nums[i] = nums[j]
            i += 1
            j += 1
        while i < len(nums):
            nums[i] = 0
            i += 1

# my second improved solution
class Solution:
    def moveZeroes(self, nums: 'List[int]') -> 'None':
        """
        Do not return anything, modify nums in-place instead.
        """
        i, j = 0, 0
        while (j < len(nums)):
            if nums[j] != 0:
                nums[i] = nums[j]
                i += 1
            j += 1
        for k in range(i, len(nums)):
            nums[k] = 0
```

第一个方法想复杂了，其实就是双指针，第一个指针指向要赋值的index，第二个指针永远指向非0index，最后剩下的没赋值的赋值为0即可

## 349. Intersection of Two Arrays ##

Given two arrays, write a function to compute their intersection.

Example 1:

Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2]

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        set1 = set()
        result = set()
        for num in nums1:
            set1.add(num)
        for num in nums2:
            if num in set1:
                result.add(num)
        return list(result)
```

## 350. Intersection of Two Arrays II ##

```java
/*************************************************************************************** 
 *
 * Given two arrays, write a function to compute their intersection.
 * 
 * Example:
 * Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2, 2].
 * 
 * Note:
 * Each element in the result should appear as many times as it shows in both arrays.
 * The result can be in any order.
 * 
 * Follow up:
 * What if the given array is already sorted? How would you optimize your algorithm?
 * What if nums1's size is small compared to num2's size? Which algorithm is better?
 * What if elements of nums2 are stored on disk, and the memory is limited such that you
 * cannot load all elements into the memory at once?
 * 
 ***************************************************************************************/
 
 /* Solution
  * --------
  *
  * Follow up:
  * 
  * 1)If the given array is already sorted we can skip the sorting.
  * 
  * 2)If nums1 is significantly smaller than nums2 we can only sort nums1 and then binary
  * search every element of nums2 in nums1 with a total complexity of (MlogN) or if nums2
  * is already sorted we can search every element of nums1 in nums2 in O(NlogM)
  *
  * 3)Just like 2), we can search for every element in nums2, thus having an online
  * algorithm.
  */

  ```python
  class Solution:
    def intersect(self, nums1: 'List[int]', nums2: 'List[int]') -> 'List[int]':
        dic = {}
        result = []
        for num in nums1:
            dic[num] = dic.get(num, 0) + 1
        for num in nums2:
            if num in dic and dic[num] > 0:
                dic[num] -= 1
                result.append(num)
        return result
```

用python十分简洁，利用dic

## 414. Third Maximum Number ##

Given a non-empty array of integers, return the third maximum number in this array. If it does not exist, return the maximum number. The time complexity must be in O(n).

```python
class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        v = [float('-Inf'), float('-Inf'), float('-Inf')]
        for num in nums:
            if num not in v:
                if num > v[0]: 
                    v = [num, v[0], v[1]]
                elif num > v[1]:
                    v = [v[0], num, v[1]]
                elif num > v[2]:
                    v = [v[0], v[1], num]
        if float('-Inf') in v:
            return v[0]
        else:
            return v[2]
```

核心在于用一个数组维持最大的三个数，不断更新

## 448. Find All Numbers Disappeared in an Array ##

Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.

Find all the elements of [1, n] inclusive that do not appear in this array.

Could you do it without extra space and in O(n) runtime? You may assume the returned list does not count as extra space.

```python
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        for i, num in enumerate(nums):
            absNum = abs(num)
            if nums[absNum - 1] > 0:
                nums[absNum - 1] = -nums[absNum - 1]
        result = []
        for i, num in enumerate(nums):
            if num > 0:
                result.append(i + 1)
        return result
```

## 485. Max Consecutive Ones ##

Given a binary array, find the maximum number of consecutive 1s in this array

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        maxNum = 0
        cnt = 0
        for num in nums:
            if num == 1:
                cnt += 1
                if cnt > maxNum:
                    maxNum = cnt
            else:
                cnt = 0
        return maxNum
```

## 532. K-diff Pairs in an Array ##

Given an array of integers and an integer k, you need to find the number of unique k-diff pairs in the array. Here a k-diff pair is defined as an integer pair (i, j), where i and j are both numbers in the array and their absolute difference is k.

```python
# Naive solution
class Solution:
    def findPairs(self, nums: List[int], k: int) -> int:
        setNums = set(nums)
        if k == 0:
            return sum(v>1 for v in collections.Counter(nums).values())
        nums = sorted(list(setNums))
        cnt = 0
        for i in range(0, len(nums) - 1, 1):
            for j in range(i+1, len(nums), 1):
                if nums[j] - nums[i] == k:
                    cnt += 1
                elif nums[j] - nums[i] > k:
                    break
        return cnt
```

```python
# Improved solution
class Solution:
    def findPairs(self, nums: List[int], k: int) -> int:
        dic = {}
        for num in nums:
            dic[num] = dic.get(num, 0) + 1
        cnt = 0
        if k == 0:
            for num in dic:
                if dic[num] > 1:
                    cnt += 1
        elif k > 0:
            for num in dic:
                if num + k in dic:
                    cnt += 1
        else:
            return 0
        return cnt
```

数组还是要想到hashmap

