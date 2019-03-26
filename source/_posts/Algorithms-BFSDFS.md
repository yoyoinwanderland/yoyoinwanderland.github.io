---
title: Algorithms - 2. BFS/ DFS
date: 2018-02-18 18:04:15
tags: 
- Coding Interviews
category: 
- 时习之
- Algorithms
description: Coding exercises - BFS/ DFS
---

## BFS Iterative

### Easy

#### Symmetric Tree

Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

For example, this binary tree `[1,2,2,3,4,4,3]` is symmetric:

```
    1
   / \
  2   2
 / \ / \
3  4 4  3

```

But the following `[1,2,2,null,3,null,3]` is not:

```
    1
   / \
  2   2
   \   \
   3    3
```

    def isSymmetric(self, root):
    	"""
        :type root: TreeNode
        :rtype: bool
        """
        # recursive
        if not root:
            return True
        
        #bfs iterative
        visited, queue = set(), [(root.left, root.right)]
        while queue:
            left, right = queue.pop(0)
            if left is None and right is None:
                continue
            if (left is None or right is None) or (left.val != right.val):
                return False
            queue.append((left.left, right.right))
            queue.append((left.right, right.left))
        
        return True
    def isSymmetric(self, root):
    	"""
        :type root: TreeNode
        :rtype: bool
        """
        # recursive
        if not root:
            return True
        
        return self.isMirror(root.left, root.right)
    
    def isMirror(self, left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        
        if left.val == right.val:
            inner = self.isMirror(left.left, right.right)
            outer = self.isMirror(left.right, right.left)
            if inner and outer:
                return True
            else:
                return False
        else:
            return False
        if not root:
            return True
#### Nested List Weight Sum

Given a nested list of integers, return the sum of all integers in the list weighted by their depth.

Each element is either an integer, or a list -- whose elements may also be integers or other lists.

**Example 1:**
Given the list `[[1,1],2,[1,1]]`, return **10**. (four 1's at depth 2, one 2 at depth 1)

**Example 2:**
Given the list `[1,[4,[6]]]`, return **27**. (one 1 at depth 1, one 4 at depth 2, and one 6 at depth 3; 1 + 4*2 + 6*3 = 27)

    def depthSum(self, nestedList):
        """
        :type nestedList: List[NestedInteger]
        :rtype: int
        """
        
        depth, acc_sum = 1, 0
        while nestedList:
            nextList = []
            for x in nestedList:
                if x.isInteger():
                    acc_sum += depth*x.getInteger()
                else:
                    for i in x.getList():
                        nextList.append(i)
            
            depth += 1
            nestedList = nextList
        
        return acc_sum
#### Nested List Weight Sum II

Given a nested list of integers, return the sum of all integers in the list weighted by their depth.

Each element is either an integer, or a list -- whose elements may also be integers or other lists.

Different from the [previous question](https://leetcode.com/problems/nested-list-weight-sum/) where weight is increasing from root to leaf, now the weight is defined from bottom up. i.e., the leaf level integers have weight 1, and the root level integers have the largest weight.

**Example 1:**
Given the list `[[1,1],2,[1,1]]`, return **8**. (four 1's at depth 1, one 2 at depth 2)

**Example 2:**
Given the list `[1,[4,[6]]]`, return **17**. (one 1 at depth 3, one 4 at depth 2, and one 6 at depth 1; 1*3 + 4*2 + 6*1 = 17)



    def depthSumInverse(self, nestedList):
    	"""
        :type nestedList: List[NestedInteger]
        :rtype: int
        """
        w_sums = []
        level = 1
        while nestedList:
            w_sum = 0
            nextList = []
            for i in nestedList:
                if i.isInteger():
                    w_sum += i.getInteger()
                else:
                    nextList += i.getList()
            nestedList = nextList
            level += 1
            w_sums.append(w_sum)
        
        return sum([ (i+1)*s for i,s in enumerate(w_sums[::-1])])
### Medium

#### Binary Tree Level Order Traversal

Given a binary tree, return the *level order* traversal of its nodes' values. (ie, from left to right, level by level).

For example:
Given binary tree `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7

```

return its level order traversal as:

```
[
  [3],
  [9,20],
  [15,7]
]
```



    def levelOrder(self, root):
    	"""
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        result, current = [], [root]
        while root and current:
            result.append([i.val for i in current if i])
            current = [kid for i in current for kid in (i.left, i.right) if kid]
        
        return result
#### Binary Tree Right Side View

Given a binary tree, imagine yourself standing on the *right* side of it, return the values of the nodes you can see ordered from top to bottom.

For example:
Given the following binary tree,

```
   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---

```

You should return `[1, 3, 4]`.

    def rightSideView(self, root):
    	"""
        :type root: TreeNode
        :rtype: List[int]
        """
        ans, levels = [], [root]
        while root and levels:
            ans.append(levels[-1].val)
            levels = [kid for n in levels for kid in (n.left, n.right) if kid]
        return ans
#### Binary Tree Zigzag Level Order Traversal

Given a binary tree, return the *zigzag level order* traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

For example:
Given binary tree `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7

```

return its zigzag level order traversal as:

```
[
  [3],
  [20,9],
  [15,7]
]
```

    def zigzagLevelOrder(self, root):
    	"""
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        ans, levels = [], [root]
        left = True
        
        while root and levels:
            next_ans = [i.val for i in levels]
            if not left:
                next_ans = next_ans[::-1]
            
            ans.append(next_ans)
            levels = [child for i in levels for child in (i.left, i.right) if child]
            left = not left
        return ans
####  Find Largest Value in Each Tree Row

You need to find the largest value in each row of a binary tree.

**Example:**

```
Input: 

          1
         / \
        3   2
       / \   \  
      5   3   9 

Output: [1, 3, 9]
```



    def largestValues(self, root):
    	"""
        :type root: TreeNode
        :rtype: List[int]
        """
        ans, levels = [], [root]
        
        while root and levels:
            ans.append(max([i.val for i in levels]))
            levels = [kid for i in levels for kid in (i.left, i.right) if kid]
        
        return ans
#### Perfect Squares

Given a positive integer *n*, find the least number of perfect square numbers (for example, `1, 4, 9, 16, ...`) which sum to *n*.

For example, given *n* = `12`, return `3` because `12 = 4 + 4 + 4`; given *n* = `13`, return `2` because `13 = 4 + 9`.



    def numSquares(self, n):
    	"""
        :type n: int
        :rtype: int
        """
        ceil = int(math.sqrt(n))
        candidates = {i**2 for i in range(1, ceil+1)}
        sums = {n}
        cnt = 0
        
        while True:
            cnt += 1
            if candidates & sums:
                return cnt
            else:
                sums.update({s - i for s in sums for i in candidates})
#### Word Ladder

Given two words (*beginWord* and *endWord*), and a dictionary's word list, find the length of shortest transformation sequence from *beginWord* to *endWord*, such that:

1. Only one letter can be changed at a time.
2. Each transformed word must exist in the word list. Note that *beginWord* is *not* a transformed word.

For example,

Given:
*beginWord* = `"hit"`
*endWord* = `"cog"`
*wordList* = `["hot","dot","dog","lot","log","cog"]`

As one shortest transformation is `"hit" -> "hot" -> "dot" -> "dog" -> "cog"`,
return its length `5`.

**Note:**

- Return 0 if there is no such transformation sequence.
- All words have the same length.
- All words contain only lowercase alphabetic characters.
- You may assume no duplicates in the word list.
- You may assume *beginWord* and *endWord* are non-empty and are not the same.



    def ladderLength(self, beginWord, endWord, wordList):
    	"""
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        wordList = set(wordList)
        wordList.discard(beginWord)
        
        ladder = 1
        front, back = {beginWord}, {endWord}
        
        while front:
            ladder += 1
            front = wordList & {word[:i] + char + word[i+1:] for word in front for i in xrange(len(word)) for char in 'abcdefghijklmnopqrstuvwxyz' if word[i] != char}
            if front & back:
                return ladder
            else:
                wordList -= front
        return 0


## DFS Recursive

### Easy

#### Maximum Depth of Binary Tree

Given a binary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

For example:
Given binary tree `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

return its depth = 3.

    def maxDepth(self, root):
    	"""
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
### Medium

#### Validate Binary Search Tree

Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:

- The left subtree of a node contains only nodes with keys **less than** the node's key.
- The right subtree of a node contains only nodes with keys **greater than** the node's key.
- Both the left and right subtrees must also be binary search trees.

**Example 1:** True

```
    2
   / \
  1   3
```

**Example 2:** False

```
    1
   / \
  2   3
```

    def isValidBST(self, root, left = float('-inf'), right = float('inf')):
    	"""
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        
        if root.val <= left or root.val >= right:
            return False
        
        return self.isValidBST(root.left, left,  root.val) and self.isValidBST(root.right, root.val, right)
#### Find Leaves of Binary Tree

Given a binary tree, collect a tree's nodes as if you were doing this: Collect and remove all leaves, repeat until the tree is empty.

**Example:**
Given binary tree 

```
          1
         / \
        2   3
       / \     
      4   5    

```

Returns `[4, 5, 3], [2], [1]`.

**Explanation:**

1. Removing the leaves `[4, 5, 3]` would result in this tree:

```
          1
         / 
        2          
```

2. Now removing the leaf `[2]` would result in this tree:

```
          1          
```

3. Now removing the leaf `[1]` would result in the empty tree:

```
          []         
```

Returns `[4, 5, 3], [2], [1]`.

```
# reference: 
# https://leetcode.com/problems/find-leaves-of-binary-tree/discuss/83851/Silly-3-liner...
def findLeaves(self, root):
    if not root: return []
    kids = map(self.findLeaves, (root.left, root.right))
    return map(lambda l, r: (l or []) + (r or []), *kids) + [[root.val]]
```



#### Number of Islands

Given a 2d grid map of `'1'`s (land) and `'0'`s (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

***Example 1:***

```
11110
11010
11000
00000
```

Answer: 1

***Example 2:***

```
11000
11000
00100
00011
```

Answer: 3

    def numIslands(self, grid):
    	"""
        :type grid: List[List[str]]
        :rtype: int
        """
        return sum(self.sink(grid, i, j) for i in range(len(grid)) for j in range(len(grid[i])))
    
    def sink(self, grid, i, j):
        if 0 <= i < len(grid) and 0 <= j < len(grid[i]) and grid[i][j] == '1':
            grid[i][j] = 0
            self.sink(grid, i-1, j)
            self.sink(grid, i+1, j)
            self.sink(grid, i, j-1)
            self.sink(grid, i, j+1)
            return 1
        return 0