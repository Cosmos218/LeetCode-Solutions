# LeetCode #

<!-- TOC -->

- [LeetCode](#leetcode)
    - [1.Two Sum](#1two-sum)
    - [26. Remove Duplicates from Sorted Array](#26-remove-duplicates-from-sorted-array)
    - [122. Best Time to Buy and Sell Stock II](#122-best-time-to-buy-and-sell-stock-ii)
    - [136. Single Number](#136-single-number)
    - [350. Intersection of Two Arrays II](#350-intersection-of-two-arrays-ii)
    - [66. Plus One](#66-plus-one)

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