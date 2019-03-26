---
title: Algorithms - 1. Array
date: 2018-02-17 15:47:14
tags: 
- Coding Interviews
category: 
- 时习之
- Algorithms
description: Coding exercises - Arrays
---



## Looping

### Easy

#### 1 Two Sum

Given an array of integers, return **indices** of the two numbers such that they add up to a specific target.

You may assume that each input would have **exactly** one solution, and you may not use the *same* element twice.

```
def twoSum(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
	sols = {}
    for i in xrange(len(nums)):
    	if nums[i] in sols:
   			 return i, sols[nums[i]]
    	else:
    		sols[target - nums[i]] = i
    return None
```



#### 243 Shortest Word Distance

Given a list of words and two words *word1* and *word2*, return the shortest distance between these two words in the list.

For example,
Assume that words = `["practice", "makes", "perfect", "coding", "makes"]`.

Given *word1* = `“coding”`, *word2* = `“practice”`, return 3.
Given *word1* = `"makes"`, *word2* = `"coding"`, return 1.

**Note:**
You may assume that *word1* **does not equal to** *word2*, and *word1* and *word2* are both in the list.

```
def shortestDistance(self, words, word1, word2):
    """
    :type words: List[str]
    :type word1: str
    :type word2: str
    :rtype: int
    """
    index1 = index2 = dist = len(words)
    for i, w in enumerate(words):
    if w == word1:
        index1 = i
        dist = min(dist, abs(index2 - index1))
    elif w == word2:
        index2 = i
        dist = min(dist, abs(index2 - index1))
	return dist
```



#### 66. Plus One

Given a non-negative integer represented as a **non-empty** array of digits, plus one to the integer.

You may assume the integer do not contain any leading zero, except the number 0 itself.

The digits are stored such that the most significant digit is at the head of the list.

```
def plusOne(self, digits):
    """
    :type digits: List[int]
    :rtype: List[int]
    """
  
 	if len(digits) == 0:
  		return [1]

	d = digits[-1] + 1
	if d < 10:
        digits[-1] = d
        return digits
    else:
        return self.plusOne(digits[:-1]) + [0]
```

```
def plusOne(self, digits):
    """
    :type digits: List[int]
    :rtype: List[int]
    """
    if len(digits) == 0:
        return [1]
    else:
        digits = reduce(lambda x,y : 10*x + y , digits) + 1
        return map(int, str(digits))
```



#### 605 Can Place Flowers

Suppose you have a long flowerbed in which some of the plots are planted and some are not. However, flowers cannot be planted in adjacent plots - they would compete for water and both would die.

Given a flowerbed (represented as an array containing 0 and 1, where 0 means empty and 1 means not empty), and a number **n**, return if **n** new flowers can be planted in it without violating the no-adjacent-flowers rule.

**Example 1:**

```
Input: flowerbed = [1,0,0,0,1], n = 1
Output: True
```

**Example 2:**

```
Input: flowerbed = [1,0,0,0,1], n = 2
Output: False
```

**Note:**

1. The input array won't violate no-adjacent-flowers rule.
2. The input array size is in the range of [1, 20000].
3. **n** is a non-negative integer which won't exceed the input array size.

```
def canPlaceFlowers(self, flowerbed, n):
    """
    :type flowerbed: List[int]
    :type n: int
    :rtype: bool
    """
    for i, x in enumerate(flowerbed):
    	if n <= 0:
    		return True

		if (not x) and (i == 0 or not flowerbed[i-1]) and (i == 			len(flowerbed) - 1 or not flowerbed[i+1]):
			n -= 1
			flowerbed[i] = 1

	return n <= 0
```



### Medium

#### 162 Find Peak Element

A peak element is an element that is greater than its neighbors.

Given an input array where `num[i] ≠ num[i+1]`, find a peak element and return its index.

The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.

You may imagine that `num[-1] = num[n] = -∞`.

For example, in array `[1, 2, 3, 1]`, 3 is a peak element and your function should return the index number 2.

    def findPeakElement(self, nums):
    	"""
        :type nums: List[int]
        :rtype: int
        """
        for i in xrange(len(nums)):
            if (i == 0 or nums[i] > nums[i-1]) and (i == len(nums) - 1 or nums[i] > nums[i+1]):
                return i
#### 73 Set Matrix Zeroes

Given a *m* x *n* matrix, if an element is 0, set its entire row and column to 0. Do it in place.

    def setZeroes(self, matrix):
    	"""
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        if not matrix or not matrix[0]:
            return
        
        rows = len(matrix)
        cols = len(matrix[0])
        row_zeros = []
        col_zeros = []
        for i in xrange(rows):
            for j in xrange(cols):
                if matrix[i][j] == 0:
                    row_zeros.append(i)
                    col_zeros.append(j)
        
        for i in row_zeros:
            matrix[i] = [0]*cols
        
        for i in xrange(rows):
            for j in col_zeros:
                matrix[i][j] = 0
#### 238 Product of Array Except Self

Given an array of *n* integers where *n* > 1, `nums`, return an array `output` such that `output[i]` is equal to the product of all the elements of `nums` except `nums[i]`.

Solve it **without division** and in O(*n*).

For example, given `[1,2,3,4]`, return `[24,12,8,6]`.

**Follow up:**
Could you solve it with constant space complexity? (Note: The output array **does not** count as extra space for the purpose of space complexity analysis.)



    def productExceptSelf(self, nums):
    	"""
        :type nums: List[int]
        :rtype: List[int]
        """
        n = len(nums)
        p = 1
        output = []
        
        for num in nums:
            output.append(p)
            p *= num
        
        p = 1
        for j in xrange(n-1, -1, -1):
            output[j] *= p
            p *= nums[j]
        
        return output


#### 78 Subsets

Given a set of **distinct** integers, *nums*, return all possible subsets (the power set).

**Note:** The solution set must not contain duplicate subsets.

For example,
If **nums** = `[1,2,3]`, a solution is:

```
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
```

    def subsets(self, nums):
    	"""
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ans = [[]]
        for i in nums:
            ans = ans + [l + [i] for l in ans]
        return ans
#### 280 Wiggle Sort

Given an unsorted array `nums`, reorder it **in-place** such that `nums[0] <= nums[1] >= nums[2] <= nums[3]...`.

For example, given `nums = [3, 5, 2, 1, 6, 4]`, one possible answer is `[1, 6, 2, 5, 3, 4]`.



    def wiggleSort(self, nums):
    	"""
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        for i in range(len(nums) - 1):
            nums[i:i+2] = sorted(nums[i:i+2], reverse = i%2)
#### 48 Rotate Image

You are given an *n* x *n* 2D matrix representing an image.

Rotate the image by 90 degrees (clockwise).

**Note:**
You have to rotate the image **in-place**, which means you have to modify the input 2D matrix directly. **DO NOT** allocate another 2D matrix and do the rotation.

**Example 1:**

```
Given input matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

rotate the input matrix in-place such that it becomes:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]

```

**Example 2:**

```
Given input matrix =
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 

rotate the input matrix in-place such that it becomes:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]
```



    def rotate(self, matrix):
    	"""
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        if matrix:
            matrix[:] = list(zip(*matrix[::-1]))
#### 54 Spiral Matrix

Given a matrix of *m* x *n* elements (*m* rows, *n* columns), return all elements of the matrix in spiral order.

For example,
Given the following matrix:

```
[[ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]]
```

You should return `[1,2,3,6,9,8,7,4,5]`

    def spiralOrder(self, matrix):
    	"""
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if not matrix or not matrix[0]:
            return matrix
        
        return  list(matrix.pop(0)) + self.spiralOrder(zip(*matrix)[::-1])
#### 15 Three Sum

Given an array *S* of *n* integers, are there elements *a*, *b*, *c* in *S* such that *a* + *b* + *c* = 0? Find all unique triplets in the array which gives the sum of zero.

**Note:** The solution set must not contain duplicate triplets.

```
For example, given array S = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```

    def threeSum(self, nums):
    	"""
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) < 3:
            return []
        
        if len(nums) == 3:
            return [] if sum(nums) != 0 else [nums]    
        
        nums = sorted(nums)
        solutions = []
        for i in range(len(nums) - 1):
            if nums[i] > 0 :
                return solutions
            if i > 0 and nums[i] == nums[i-1]:
                continue
            
            sols = self.twoSum(nums[i+1:], -nums[i])
            solutions.extend(sols)
        
        return solutions
    
    def twoSum(self, nums, target):
        sols = {}
        sub_dict = {}
        for num in nums:
            if num in sub_dict:
                sols[num] =  target-num
            else:
                sub_dict[target - num] = 1
    
        sols = [[-target, key, val] for key, val in sols.items()]
        return sols
## Two Pointers

### Easy

#### 283 Move Zeros

Given an array `nums`, write a function to move all `0`'s to the end of it while maintaining the relative order of the non-zero elements.

For example, given `nums = [0, 1, 0, 3, 12]`, after calling your function, `nums` should be `[1, 3, 12, 0, 0]`.

**Note**:

1. You must do this **in-place** without making a copy of the array.
2. Minimize the total number of operations.

```
def moveZeroes(self, nums):
    zero = 0  # records the position of "0"
    for i in xrange(len(nums)):
        if nums[i] != 0:
            nums[i], nums[zero] = nums[zero], nums[i]
            zero += 1
```

#### 88 Merge Sorted Array

Given two sorted integer arrays *nums1* and *nums2*, merge *nums2* into *nums1* as one sorted array.

**Note:**
You may assume that *nums1* has enough space (size that is greater or equal to *m* + *n*) to hold additional elements from *nums2*. The number of elements initialized in *nums1* and *nums2* are *m* and *n* respectively.

    def merge(self, nums1, m, nums2, n):
       """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        while m > 0 and n > 0:
            if nums1[m-1] > nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -=1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        if n > 0:
            nums1[:n] = nums2[:n]

### Medium

#### 75 Sort Colors

Given an array with *n* objects colored red, white or blue, sort them so that objects of the same color are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

    def sortColors(self, nums):
    	"""
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        i = j = 0
        for k in xrange(len(nums)):
            n = nums[k]
            nums[k] = 2
            if n < 2:
                nums[j] = 1
                j += 1
            if n == 0:
                nums[i] = 0
                i += 1
## Dynamic Programming

### Easy

#### 53 Maximum Subarray 

Find the contiguous subarray within an array (containing at least one number) which has the largest sum.

For example, given the array `[-2,1,-3,4,-1,2,1,-5,4]`,
the contiguous subarray `[4,-1,2,1]` has the largest sum = `6`.

```
def maxSubArray(self, nums):
    """
    :type nums: List[int]
    :rtype: int

    Example: [-2,1,-3,4,-1,2,1,-5,4]
    1. if added_sum  + current number < current sum: 
    	abandon added_sum 
    2. keep track the largest sum in previous sublists
    """
    if not nums:
    	return 0
    if len(nums) == 1:
    	return nums[0]

    prev_max = current_max = nums[0]
    for n in nums[1:]:
      current_max = max(current_max + n, n)
      prev_max = max(current_max, prev_max)
    return prev_max
```



#### 121 Best Time to Buy and Sell Stock

Say you have an array for which the *i*th element is the price of a given stock on day *i*.

If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.

**Example 1:**
Input: [7, 1, 5, 3, 6, 4]
Output: 5

max. difference = 6-1 = 5 (not 7-1 = 6, as selling price needs to be larger than buying price)


**Example 2:**
Input: [7, 6, 4, 3, 1]
Output: 0

In this case, no transaction is done, i.e. max profit = 0.

    def maxProfit(self, prices):
    	"""
        :type prices: List[int]
        :rtype: int
        """
    
        min_prices = float('inf')
        max_profit = 0
        
        for i in prices:
            min_prices = min(i, min_prices)
            max_profit = max( i - min_prices, max_profit)
        return max_profit



#### 76 Min Cost Climbing Stairs

On a staircase, the `i`-th step has some non-negative cost `cost[i]` assigned (0 indexed).

Once you pay the cost, you can either climb one or two steps. You need to find minimum cost to reach the top of the floor, and you can either start from the step with index 0, or the step with index 1.

**Example 1:**
Input: cost = [10, 15, 20]
Output: 15
Explanation: Cheapest is start on cost[1], pay that cost and go to the top.


**Example 2:**
Input: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
Output: 6
Explanation: Cheapest is start on cost[0], and only step on 1s, skipping cost[3].


**Note:**

1. `cost` will have a length in the range `[2, 1000]`.
2. Every `cost[i]` will be an integer in the range `[0, 999]`.



    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        
        # if no stairs / only one stair, no cost by climbing 2 steps
        
        if len(cost) < 2:
            return 0
        
        # by reaching the second to last, or the last of the array, we can finish the whole journey
        # we proceed like this:
        # step 1, step 2
        #         step 2, step 3
        #                 step 3, step 4
        # on each of stairs, we try to find the minimal cost to reach there
        # the final minimum cost is the min ( cost_of_two_steps_away, cost_of_one_step_away)
        
        two_steps_away_cost, one_step_away_cost = cost[0], cost[1]
        
        ## i is the step to reach
        for i in range(2, len(cost)):
            two_steps_away_cost, one_step_away_cost = one_step_away_cost, min(two_steps_away_cost, one_step_away_cost) + cost[i]
        return min(two_steps_away_cost, one_step_away_cost)
### Medium

#### 152 Maximum Product Subarray

Find the contiguous subarray within an array (containing at least one number) which has the largest product.

For example, given the array `[2,3,-2,4]`,
the contiguous subarray `[2,3]` has the largest product = `6`.

    def maxProduct(self, nums):
    	"""
        :type nums: List[int]
        :rtype: int
        """
        prev_min = prev_max = max_val = nums[0]
        
        for i in nums[1:]:
            prev_min, prev_max = min(prev_min*i, prev_max*i, i), max(prev_max*i, prev_min*i, i)
            max_val = max(max_val, prev_max)
        return max_val


## Binary Search

### Medium

#### 34 Search for a Range

Given an array of integers sorted in ascending order, find the starting and ending position of a given target value.

Your algorithm's runtime complexity must be in the order of *O*(log *n*).

If the target is not found in the array, return `[-1, -1]`.

For example,
Given `[5, 7, 7, 8, 8, 10]` and target value 8,
return `[3, 4]`.
```
    def searchRange(self, nums, target):
    	"""
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        if not nums:
            return -1, -1
        start = bisect.bisect_left(nums, target)
        if start >= len(nums) or nums[start] != target :
            return -1, -1
        
        end = bisect.bisect_right(nums, target, lo = start)
        return start, end - 1
```
```
    def searchRange(self, nums, target):
    	if not nums:
            return [-1,-1]
        start = self.binarySearch(nums, target-0.5)
        if nums[start] != target:
            return [-1, -1]
        nums.append(float('inf'))
        end = self.binarySearch(nums, target+0.5)-1
        return [start, end]
    def binarySearch(self, arr, target):
        start, end = 0, len(arr)-1
        while start < end:
            mid = (start+end)//2
            if target < arr[mid]:
                end = mid
            else:
                start = mid+1
        return start
```

#### 33 Search in Rotated Sorted Array

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., `0 1 2 4 5 6 7` might become `4 5 6 7 0 1 2`).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.

    def search(self, nums, target):
    	"""
        :type nums: List[int]
        :type target: int
        :rtype: int
        """ 
        if not nums:
            return -1
        
        if len(nums) == 1:
            return int(nums[0] == target) - 1
        
        left, right  = 0, len(nums) - 1
        
        while left < right:
            mid = (left + right)/ 2
            if nums[mid] == target:
                return mid
            elif nums[left] <= nums[mid] < target or target < nums[left] <= nums[mid] or nums[mid] < target < nums[left]:
                left = mid + 1
            else:
                right = mid
    
        return left if target == nums[left] else -1
## Mark Occurance Using Array

### Easy

#### 448 Find All Numbers Disappeared in an Array

Given an array of integers where 1 ≤ a[i] ≤ *n* (*n* = size of array), some elements appear twice and others appear once.

Find all the elements of [1, *n*] inclusive that do not appear in this array.

Could you do it without extra space and in O(*n*) runtime? You may assume the returned list does not count as extra space.

**Example:**

Input: `[4,3,2,7,8,2,3,1]`
Output: `[5,6]`

    def findDisappearedNumbers(self, nums):
    	"""
        :type nums: List[int]
        :rtype: List[int]
        """
        # return list(set(range(1, len(nums) + 1)) - set(nums))
        
        for n in nums:
            ix = int(n) - 1
            nums[ix] += 0.3
        
        return [i + 1 for i, n in enumerate(nums) if n - int(n) == 0]
## DFS

### Medium

#### 79 Word Search

Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

For example, given **board** =

```
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

```

word = "ABCCED", -> returns true

word = "SEE", -> returns true

word = "ABCB", -> returns false



    def exist(self, board, word):
    	"""
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        if not board or not board[0] or not word:
            return False
        
        for i in xrange(len(board)):
            for j in xrange(len(board[i])):
                if self.dfs(board, word, i, j):
                    return True
        
        return False
        
    def dfs(self, board, word, i, j):
        if not word:
            return True
        if board[i][j] != word[0]:
            return False
        if not word[1:]:
            return True
        
        board[i][j] = "#"
        if i-1 >= 0 and self.dfs(board, word[1:], i-1, j):
            return True 
        if j-1 >= 0 and self.dfs(board, word[1:], i, j-1):
            return True
        if i+1 < len(board) and self.dfs(board, word[1:], i+1, j):
            return True 
        if j+1 < len(board[i]) and self.dfs(board, word[1:], i, j+1):
            return True
        
        board[i][j] = word[0]
        return False
