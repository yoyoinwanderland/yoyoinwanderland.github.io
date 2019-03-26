---
title: Algorithms - 3. Backtracking
date: 2018-02-19 17:44:59
tags: 
- Coding Interviews
category: 
- 时习之
- Algorithms
description: Coding exercises - backtracking
---

All problems are medium-level.

## Iterate and Append Solutions

###  Subsets

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
            ans += [l + [i] for l in ans]
        return ans


### Gray Code



The gray code is a binary numeral system where two successive values differ in only one bit.

Given a non-negative integer *n* representing the total number of bits in the code, print the sequence of gray code. A gray code sequence must begin with 0.

For example, given *n* = 2, return `[0,1,3,2]`. Its gray code sequence is:

```
00 - 0
01 - 1
11 - 3
10 - 2

```

**Note:**
For a given *n*, a gray code sequence is not uniquely defined.

For example, `[0,2,3,1]` is also a valid gray code sequence according to the above definition.

For now, the judge is able to judge based on one instance of gray code sequence. Sorry about that.

    def grayCode(self, n):
    	"""
        :type n: int
        :rtype: List[int]
        """
        ans = [0]
        for i in range(n):
            ans +=  [a + 2**i for a in ans[::-1]]
        return ans


### Letter Combinations of a Phone Number

Given a digit string, return all possible letter combinations that the number could represent.

A mapping of digit to letters (just like on the telephone buttons) is given below.

![img](http://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Telephone-keypad2.svg/200px-Telephone-keypad2.svg.png)

```
Input:Digit string "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].

```

**Note:**
Although the above answer is in lexicographical order, your answer could be in any order you want.

    def letterCombinations(self, digits):
    	"""
        :type digits: str
        :rtype: List[str]
        """
        sols = []
        digits = digits.replace("1","").replace("0","")
        
        if not digits:
            return sols
        
        digits_list = list(digits)
        while digits_list:
            digit, digits_list  = digits_list[0], digits_list[1:]
            sols = self.mapping(digit, sols)
        return sols
    
    def mapping(self, digit, sols):
        """
        :type digits_list: 
        """
        letter_map = {"2":"abc", "3": "def", "4":"ghi", "5":"jkl", "6":"mno", "7":"pqrs", "8":"tuv", "9":"wxyz"}
    
        if not sols:
            sols = list(letter_map[digit])
        else:
            sols = [i+m for i in sols for m in letter_map[digit]]
        return sols


## Iterate While Eliminate Duplicates

### Permutations II

Given a collection of numbers that might contain duplicates, return all possible unique permutations.

For example,
`[1,1,2]` have the following unique permutations:

```
[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]
```

    def permuteUnique(self, nums):
    	"""
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        sols = [[]]
    
        for n in nums:
            sols = [s[:pos] + [n] + s[pos:] 
                    for s in sols 
                    for pos in xrange((s + [n]).index(n) + 1)]
                    
        return sols


### Factor Combinations

Numbers can be regarded as product of its factors. For example,

```
8 = 2 x 2 x 2;
  = 2 x 4.

```

Write a function that takes an integer *n* and return all possible combinations of its factors.

**Note:** 

1. You may assume that *n* is always positive.
2. Factors should be greater than 1 and less than *n*.

**Examples: **
input: `1` output: []

input:  37 output:  []

input:  12  output:

```
[
  [2, 6],
  [2, 2, 3],
  [3, 4]
]

```

input:  32 output:

```
[
  [2, 16],
  [2, 2, 8],
  [2, 2, 2, 4],
  [2, 2, 2, 2, 2],
  [2, 4, 4],
  [4, 8]
]
```

    def getFactors(self, n):
    	"""
        :type n: int
        :rtype: List[List[int]]
        """
        nums, sols = [(n, 2, [])], []
        
        while nums:
            n, i, factors = nums.pop()
            while i**2 <= n:
                if n%i == 0:
                    sols += [factors + [n/i, i]]
                    nums.append( (n/i, i, factors + [i]))
                i += 1
        return sols


## DFS

### Generate Parentheses

Given *n* pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

For example, given *n* = 3, a solution set is:

```
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```

    def generateParenthesis(self, n):
    	"""
        :type n: int
        :rtype: List[str]
        
        Examples: belows shows possible examples and reasons for each step
        1   (        # only starts with left p when left p and right p equals
        2   (),  ((, # can choose left or right if left
        3   ()(, (((, (()
        4   ()(), ()((, (((), (()(, (()) # run out of right p, so can't choose to go left
        5   ()()(, ()((), ((()), (()(), (())( 
        6   all
        """
        return self.dfsParenthesis(n, n, [""])
        
    def dfsParenthesis(self, left, right, sols):
        if not left and not right:
            return sols
        if not left and right:
            return self.dfsParenthesis(left, right - 1, [s + ")" for s in sols])
        if left == right:
            return self.dfsParenthesis(left-1, right, [s + "(" for s in sols])
        else:
            return self.dfsParenthesis(left-1, right, [s + "(" for s in sols]) + self.dfsParenthesis(left, right - 1, [s + ")" for s in sols])


### Word Search

Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

For example,
Given **board** =

```
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

```

"ABCCED" -> true

"SEE" -> true

"ABCB" -> false

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