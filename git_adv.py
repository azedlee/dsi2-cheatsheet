# Git Merge
"""
git pull does git fetch and git merge at the same time

git log and git reflog show you the history of your commits, reflog being more detailed


		     add files		          EDA     cleaning      docs
master ----------|----------|----------|----------|----------|----------|----------|----------|----------|
							|													   |
							|		  RFS      Descent      		   Pull		   |
modeling                    |----------|----------|----------|----------|		   |
																				   |
																				   |		flask      Pull
backend   																		   |----------|----------|

Pull request from master to get branch, then the master will merge

After git add and commit, when there is a merge conflict

git pull origin <branch>

nano file.<ext>

Once opened, there should be the merge conflict changes and the commit log.
All changes at this point is controlled by the person who pulls last.

Then git add, commit and push origin <branch>

git rebase
	with multiple commits on multiple branches, the branch gets really messy with a history of backref commits.
	with rebase, you can remove all backrefs and converge them into 1 commit into the master branch, which ideally
	cleans up all the messy history. The downside is, the history of logs in the branches are lost forever, so this
	is only good if you do not regret the history.


"""

