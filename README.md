# LeetCode #

<!-- TOC -->

- [LeetCode](#leetcode)
    - [1.Two Sum](#1two-sum)
    - [14. Longest Common Prefix](#14-longest-common-prefix)
    - [15. 3Sum](#15-3sum)
    - [26. Remove Duplicates from Sorted Array](#26-remove-duplicates-from-sorted-array)
    - [49. Group Anagrams](#49-group-anagrams)
    - [66. Plus One](#66-plus-one)
    - [73. Set Matrix Zeroes](#73-set-matrix-zeroes)
    - [88. Merge Sorted Array](#88-merge-sorted-array)
    - [122. Best Time to Buy and Sell Stock II](#122-best-time-to-buy-and-sell-stock-ii)
    - [136. Single Number](#136-single-number)
    - [278. First Bad Version(binary search)](#278-first-bad-versionbinary-search)
    - [283. Move Zeroes](#283-move-zeroes)
    - [350. Intersection of Two Arrays II](#350-intersection-of-two-arrays-ii)

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
