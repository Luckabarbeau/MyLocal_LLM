"""Deterministically build original, versioned 0097 benchmark banks.

No external datasets.  Regenerate with:
    python -m benchmarks.generate_challenge_banks_0097
The staged tests have explicit oracle answers computed during generation.
Old 0092-0096 banks are intentionally untouched.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).parent/'tasks'
R = random.Random(97122026)


def choose(correct, wrong, rotation):
    vals = [str(correct)] + [str(v) for v in wrong if str(v)!=str(correct)]
    unique = list(dict.fromkeys(vals))
    if len(unique) < 4:
        raise ValueError('Not four unique choices: '+str(vals))
    wrong = unique[1:4]
    result = wrong[:]
    index = rotation % 4
    result.insert(index, str(correct))
    return [' '+s for s in result], index


def write(filename, family, tasks, purpose):
    ids=[r['id'] for r in tasks]
    assert len(ids) == len(set(ids)) and tasks
    (ROOT/filename).write_text(json.dumps({'version':1,'family':family,'purpose':purpose,
                                           'tasks':tasks},ensure_ascii=False,indent=2)+'\n')


def math_bank():
    rows=[]
    for j in range(30):
        a=31+11*j; b=13+7*j; m=[97,101,127,257,1009][j%5]
        # Each successive probe adds a mathematical operation; a wrong answer
        # indicates the first unsupported complexity, not an internal CoT.
        trials=[
          ('addition',a+b,f'The sum of {a} and {b} equals', [a+b+1,a+b-1,a+b+3]),
          ('multiplication',a*b,f'The exact product of {a} and {b} equals', [a*b+a,a*b+b,a*b-1]),
          ('compound_arithmetic',a*b+a+b,
           f'For a={a} and b={b}, the expression a*b + a + b evaluates to',
           [a*b+a+b+1, a*b+a+b-2, a*b+a]),
          ('modular_exponentiation',pow(a, b+9, m),
           f'Using integer modular arithmetic, ({a} ** {b+9}) mod {m} equals',
           [(pow(a,b+9,m)+q)%m for q in (1,2,3)]),
          ('nested_modular', sum(pow(a+i,3, 9973) for i in range(250+j))%9973,
           f'For a={a} and n={250+j}, evaluate (sum of (a+i)^3 for i=0,...,n-1) mod 9973:',
           [(sum(pow(a+i,3,9973) for i in range(250+j))%9973+k)%9973 for k in (1,2,5)]),
          ('expert_large_integer', a*b*1_000_003 + 37*a*a + b**3,
           f'The exact integer value of {a*b}*1000003 + 37*{a}^2 + {b}^3 is',
           [a*b*1_000_003 + 37*a*a + b**3+k for k in (-1,1,10)])
        ]
        for stage,(category,answer,prompt,wrong) in enumerate(trials,1):
            choices, idx=choose(answer,wrong,j+stage)
            rows.append({'id':f'math_ladder_{j:02d}_s{stage}', 'category':category,
                         'difficulty':['basic','intermediate','intermediate','hard','expert','extreme'][stage-1],
                         'scenario_id':f'math_ladder_{j:02d}', 'stage':stage,
                         'prompt':prompt, 'choices':choices, 'answer':idx})
    return rows


PEOPLE=['Ada','Ben','Cleo','Dara','Eli','Faye','Gus','Hana','Ivan','Jules',
        'Kira','Lena','Mira','Niko','Omar','Pia','Quinn','Rhea','Sam','Tara',
        'Uma','Vera','Will','Xena','Yara','Zane']


def logic_bank():
    rows=[]
    for j in range(30):
        names=R.sample(PEOPLE,8)
        original=names[:]
        steps=[(1,4),(2,6),(0,7),(3,5),(0,2),(5,6)]
        moved=names[:]
        for a,b in steps:
            moved[a],moved[b]=moved[b],moved[a]
        final=moved[::-1]
        trial=[
            ('direct_order',original[0],f'The left-to-right queue is {", ".join(original)}. The first person is'),
            ('indexed_position',original[5],f'The left-to-right queue is {", ".join(original)}. The sixth person is'),
            ('one_swap',names[4],f'The left-to-right queue starts as {", ".join(original)}. Swap the people at positions 2 and 5 (numbering from 1). The person now second is'),
            ('multi_swap',moved[3], f'The original queue is {", ".join(original)}. Swap 1-based positions '
             '2 and 5, then 3 and 7, then 1 and 8, then 4 and 6, then 1 and 3, then 6 and 7. The person fourth is'),
            ('reverse_and_swap',final[2],f'Start with queue {", ".join(original)}. Swap 1-based positions '
             '2 with 5, 3 with 7, 1 with 8, 4 with 6, 1 with 3, and 6 with 7, in that order. '
             'Now reverse the entire queue. The third person is'),
        ]
        # above stage 3 expected after first swap is original[4]
        for stage,(typ,answer,prompt) in enumerate(trial,1):
            candidates=[answer]+[x for x in names if x!=answer][:3]
            choices,idx=choose(answer,candidates[1:],j+stage)
            rows.append({'id':f'logic_queue_{j:02d}_s{stage}', 'category':typ,
                         'difficulty':['basic','intermediate','hard','expert','extreme'][stage-1],
                         'scenario_id':f'logic_queue_{j:02d}', 'stage':stage,
                         'prompt':prompt, 'choices':choices, 'answer':idx})
    return rows


PLACES=['observatory','archive','workshop','library','hangar','courtyard','storeroom','attic',
        'boathouse','garage','laboratory','basement','greenhouse','office','market','depot',
        'gallery','museum','station','garden','tower','barn','terminal','kitchen']
OBJECTS=['red notebook','silver key','blue compass','copper ring','yellow folder',
         'wooden box','sealed letter','brass token','glass prism','green ledger']


def narrative_bank():
    rows=[]
    for j in range(36):
        item=OBJECTS[j%len(OBJECTS)]
        p=R.sample(PLACES, 6)
        keeper=PEOPLE[(j*3)%len(PEOPLE)]
        witness=PEOPLE[(j*3+1)%len(PEOPLE)]
        # Six readings over the same factual narrative. Early stages ask literal
        # facts, later ones require tracking temporal state, false reports and
        # contents of moved containers.
        context=(f'{keeper} put the {item} in the {p[0]} on Monday. '
                 f'On Tuesday {keeper} moved the {item} to the {p[1]}. '
                 f'{witness} heard an outdated message claiming the {item} was still in the {p[0]}. '
                 f'The next day {keeper} moved the {item} to the {p[2]}. '
                 f'On Thursday {keeper} placed the {item} inside a sealed satchel while in the {p[2]}. '
                 f'The satchel was carried to the {p[3]} without opening it. '
                 f'{witness} repeated the old message, but a signed log confirmed the new location. '
                 f'On Friday the satchel was transferred to the {p[4]}, still sealed. '
                 f'A written notice incorrectly guessed it was in the {p[5]}. '
                 'The verified movement log was complete and no later move took place. ')
        prompts=[
            ('direct_recall',f'At the beginning of Monday, the {item} was in the',p[0]),
            ('temporal_update',f'On Tuesday after the first move, the {item} was in the',p[1]),
            ('belief_vs_fact',f'The outdated message claimed the {item} was in the',p[0]),
            ('container_tracking',f'After the satchel was first carried away from the original packing site, the {item} was in the',p[3]),
            ('negation_and_update',f'According to the verified log, on Friday after the final transfer, the {item} was in the',p[4]),
            ('multi_hop_revision',f'Ignoring the incorrect written notice and the outdated message, the final verified location of the {item} was the',p[4]),
        ]
        for stage,(typ,question,answer) in enumerate(prompts,1):
            choices,idx=choose(answer,[s for s in p if s!=answer][:3],j+stage)
            rows.append({'id':f'narrative_ladder_{j:02d}_s{stage}',
                         'scenario_id':f'narrative_ladder_{j:02d}', 'stage':stage,
                         'category':typ,'challenge_type':typ,
                         'difficulty':['basic','intermediate','hard','hard','expert','extreme'][stage-1],
                         'prompt':context+f'Later, {keeper} began to write a report: ',
                         'recall_prompt':context+' '+question,
                         'choices':choices,'answer':idx,'anchors':[keeper,item]})
    return rows


def terminal_bank():
    rows=[]
    commands=[
      ('show_directory','Display the current directory','pwd',r'pwd'),
      ('list_hidden','List hidden files along with other files','ls -la',r'ls -(?:la|al|a)'),
      ('git_status','Show a concise Git working-tree status','git status --short',r'git status (?:--short|-s)'),
      ('branch_create','Create and switch to a new Git branch named experiment','git switch -c experiment',r'git (?:switch -c|checkout -b) experiment'),
      ('git_changes','Show unstaged changes in working files','git diff',r'git diff'),
      ('git_cached','List names of staged files','git diff --cached --name-only',r'git diff --cached --name-only'),
      ('git_head','Show the last five commits as one-line summaries','git log -5 --oneline',r'git log (?:-5 --oneline|--oneline -5)'),
      ('find_py','Find Python files recursively under the current directory',"find . -type f -name '*.py'",r"find \. -type f -name ['\"]\*\.py['\"]"),
      ('count_py','Count Python files recursively in the current directory',"find . -type f -name '*.py' | wc -l",r"find \. -type f -name ['\"]\*\.py['\"] \| wc -l"),
      ('grep_todo','Find occurrences of TODO recursively under src',"grep -Rn 'TODO' src",r"grep -Rn ['\"]TODO['\"] src"),
      ('venv','Create an isolated Python environment named .venv','python3 -m venv .venv',r'python3 -m venv \.venv'),
      ('py_compile','Check main.py for Python syntax errors without running it','python3 -m py_compile main.py',r'python3 -m py_compile main\.py'),
      ('git_branch','Show the current Git branch name','git branch --show-current',r'git branch --show-current'),
      ('git_remote','List the configured Git remote URLs','git remote -v',r'git remote -v'),
      ('git_stash','List existing Git stashes without applying them','git stash list',r'git stash list'),
      ('git_fetch','Download remote refs without merging changes','git fetch --all',r'git fetch --all'),
      ('sort_unique','Sort lines in data.txt and remove adjacent duplicate lines','sort -u data.txt',r'sort -u data\.txt'),
      ('tail_logs','Print the last 50 lines of server.log','tail -n 50 server.log',r'tail -n 50 server\.log'),
      ('disk_usage','Show human-readable available filesystem space','df -h',r'df -h'),
      ('memory_usage','Show human-readable available system RAM','free -h',r'free -h'),
      ('git_show','Show the patch of the most recent Git commit','git show HEAD',r'git show (?:HEAD)?'),
      ('git_ref','Resolve the current Git HEAD commit hash','git rev-parse HEAD',r'git rev-parse HEAD'),
      ('find_large','Find files larger than 100 megabytes under data',"find data -type f -size +100M",r'find data -type f -size \+100M'),
      ('grep_fixed','Search for the literal text a.b in notes.txt',"grep -F 'a.b' notes.txt",r"grep -F ['\"]a\.b['\"] notes\.txt"),
      ('git_worktree','List existing Git worktrees','git worktree list',r'git worktree list'),
      ('sha','Compute SHA-256 checksum of model.bin','sha256sum model.bin',r'sha256sum model\.bin'),
      ('file_type','Identify the file type of checkpoint.bin','file checkpoint.bin',r'file checkpoint\.bin'),
      ('process','Show running processes in full format','ps aux',r'ps aux'),
      ('print_env','Print PATH environment variable','printf "%s\\n" "$PATH"',r'printf .*'),
      ('list_ports','Show listening TCP sockets','ss -ltn',r'ss -ltn'),
    ]
    # Multiple differently phrased tasks per core command, with comparable
    # distractors. Harder ones require flags or pipelines; no generated shell
    # commands are ever executed in the benchmark.
    for j,(id_,goal,command,regex) in enumerate(commands):
        d1=commands[(j+7)%len(commands)][2]; d2=commands[(j+13)%len(commands)][2]
        choices, idx=choose(command,[d1,d2,'ls'],j)
        rows.append({'id':'terminal_advanced_'+id_, 'category':'git' if id_.startswith('git') else 'linux',
                     'difficulty':'expert' if any(c in id_ for c in ['count','grep','git_cached','find_large']) else 'intermediate',
                     'prompt':'# '+goal+'.\n$ ', 'choices':choices,'answer':idx,
                     'accepted_patterns':[regex]})
    return rows


def code_banks():
    py=[]
    defs=[
      ('running_total','def running_total(values: list[int]) -> list[int]:\n    """Return prefix sums, preserving order."""\n',
       [([1,2,3],[1,3,6]), ([],[]), ([-2,5,1],[-2,3,4])], 'intermediate'),
      ('median_odd','def median_odd(values: list[int]) -> int:\n    """Return the median of a nonempty odd-length list without changing it."""\n',
       [([9,2,4],4),([5],5),([11,8,3,1,7],7)], 'intermediate'),
      ('is_palindrome','def is_palindrome(text: str) -> bool:\n    """Return whether text reads the same backwards."""\n',
       [('abba',True),('abc',False),('',True)],'basic'),
      ('balanced_parentheses','def balanced_parentheses(text: str) -> bool:\n    """Check properly nested parentheses, ignoring other characters."""\n',
       [('(()())',True),(')(()',False),('(a)(b)',True),('(()',False)],'hard'),
      ('binary_search','def binary_search(values: list[int], target: int) -> int:\n    """Return index of target in sorted values, or -1 if absent."""\n',
       [(([1,3,6,9],6),2),(([1,3,6,9],4),-1),(([],0),-1)],'hard'),
      ('merge_sorted','def merge_sorted(a: list[int], b: list[int]) -> list[int]:\n    """Merge sorted lists while retaining duplicates."""\n',
       [(([1,3],[2,3]),[1,2,3,3]),(([],[1]),[1]),(([0,0],[0]),[0,0,0])],'hard'),
      ('longest_common_prefix','def longest_common_prefix(words: list[str]) -> str:\n    """Return the longest common prefix of every string."""\n',
       [(['flower','flow','flight'],'fl'),(['x'],'x'),(['a','b'],'')],'hard'),
      ('transpose','def transpose(matrix: list[list[int]]) -> list[list[int]]:\n    """Transpose a rectangular integer matrix."""\n',
       [([[1,2],[3,4]],[[1,3],[2,4]]),([[1,2,3]],[[1],[2],[3]])],'intermediate'),
      ('matrix_product','def matrix_product(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:\n    """Multiply rectangular integer matrices."""\n',
       [(([[1,2],[3,4]],[[5,6],[7,8]]),[[19,22],[43,50]]),(([[2,3]],[[4],[5]]),[[23]])],'expert'),
      ('gcd','def gcd(a: int, b: int) -> int:\n    """Return the nonnegative greatest common divisor."""\n',
       [((12,18),6),((0,5),5),((-21,14),7)],'intermediate'),
      ('prime_factors','def prime_factors(n: int) -> list[int]:\n    """Return prime factors with multiplicity in ascending order."""\n',
       [(84,[2,2,3,7]),(2,[2]),(1,[])],'expert'),
      ('sieve','def sieve(n: int) -> list[int]:\n    """List all prime numbers up to and including n."""\n',
       [(10,[2,3,5,7]),(2,[2]),(1,[])],'expert'),
      ('edit_distance','def edit_distance(a: str, b: str) -> int:\n    """Levenshtein distance with insertions, deletions and substitutions."""\n',
       [(('kitten','sitting'),3),(('','abc'),3),(('abc','abc'),0)],'extreme'),
      ('toposort','def toposort(graph: dict[str, list[str]]) -> list[str]:\n    """Return a topological ordering of a DAG; break ties lexicographically."""\n',
       [({'a':['b'],'b':[]},['a','b']),({'a':[],'b':[]},['a','b'])],'extreme'),
      ('count_islands','def count_islands(grid: list[list[int]]) -> int:\n    """Count 4-connected groups of 1 cells in a rectangular 0/1 grid."""\n',
       [([[1,0,1],[1,0,0],[0,1,1]],3),([[0,0]],0),([[1]],1)],'expert'),
      ('prefix_function','def prefix_function(text: str) -> list[int]:\n    """Compute the Knuth-Morris-Pratt prefix function."""\n',
       [('ababa',[0,0,1,2,3]),('aaaa',[0,1,2,3])],'extreme'),
      ('fibonacci_fast','def fibonacci_fast(n: int) -> int:\n    """Return F(n), with F(0)=0 and F(1)=1, using O(log n) arithmetic steps."""\n',
       [(0,0),(1,1),(20,6765),(50,12586269025)],'extreme'),
      ('determinant3','def determinant3(m: list[list[int]]) -> int:\n    """Compute determinant of a 3 by 3 integer matrix."""\n',
       [([[1,0,0],[0,1,0],[0,0,1]],1),([[1,2,3],[4,5,6],[7,8,9]],0)],'expert'),
      ('mod_inverse','def mod_inverse(a: int, m: int) -> int:\n    """Return modular inverse when gcd(a,m)=1."""\n',
       [((3,11),4),((7,19),11)],'expert'),
      ('flatten_nested','def flatten_nested(items: list) -> list:\n    """Recursively flatten lists, preserving non-list elements."""\n',
       [([1,[2,[3],4]], [1,2,3,4]),([],[])], 'hard'),
    ]
    for name,prompt,tests,difficulty in defs:
        items=[]
        for inp,out in tests:
            args=list(inp) if isinstance(inp,tuple) else [inp]
            items.append({'args':args,'expected':out})
        py.append({'id':'challenge_'+name,'category':name,'difficulty':difficulty,
                   'function':name,'prompt':prompt,'tests':items})

    cpp=[]
    defs_cpp=[
      ('running_total','#include <vector>\nstd::vector<int> running_total(const std::vector<int>& xs) {\n',
       'int main(){return running_total({1,2,3})==std::vector<int>({1,3,6}) && running_total({}).empty() ? 0:1;}', 'intermediate'),
      ('balanced_parens','#include <string>\nbool balanced_parens(const std::string& x) {\n',
       'int main(){return balanced_parens("(()())")&&!balanced_parens("(()")&&!balanced_parens(")(")?0:1;}', 'hard'),
      ('gcd_int','int gcd_int(int a, int b) {\n',
       'int main(){return gcd_int(12,18)==6&&gcd_int(0,5)==5?0:1;}', 'intermediate'),
      ('binary_search_index','#include <vector>\nint binary_search_index(const std::vector<int>& v,int target) {\n',
       'int main(){return binary_search_index({1,3,5},3)==1 && binary_search_index({1,3,5},4)==-1?0:1;}', 'hard'),
      ('merge_sorted_int','#include <vector>\nstd::vector<int> merge_sorted_int(const std::vector<int>& a,const std::vector<int>& b) {\n',
       'int main(){return merge_sorted_int({1,3},{2,3})==std::vector<int>({1,2,3,3})?0:1;}', 'hard'),
      ('prime_factor_count','int prime_factor_count(int n) {\n',
       'int main(){return prime_factor_count(12)==3&&prime_factor_count(13)==1&&prime_factor_count(1)==0?0:1;}', 'expert'),
      ('transpose_matrix','#include <vector>\nstd::vector<std::vector<int>> transpose_matrix(const std::vector<std::vector<int>>& a) {\n',
       'int main(){return transpose_matrix({{1,2},{3,4}})==std::vector<std::vector<int>>({{1,3},{2,4}})?0:1;}', 'expert'),
      ('sieve_primes','#include <vector>\nstd::vector<int> sieve_primes(int n) {\n',
       'int main(){return sieve_primes(10)==std::vector<int>({2,3,5,7}) && sieve_primes(1).empty()?0:1;}', 'expert'),
      ('count_islands_cpp','#include <vector>\nint count_islands_cpp(const std::vector<std::vector<int>>& grid) {\n',
       'int main(){return count_islands_cpp({{1,0,1},{1,0,0},{0,1,1}})==3?0:1;}', 'extreme'),
      ('edit_distance_cpp','#include <string>\nint edit_distance_cpp(const std::string& a,const std::string& b) {\n',
       'int main(){return edit_distance_cpp("kitten","sitting")==3&&edit_distance_cpp("","abc")==3?0:1;}', 'extreme'),
      ('matmul2','#include <vector>\nstd::vector<std::vector<int>> matmul2(const std::vector<std::vector<int>>& a,const std::vector<std::vector<int>>& b) {\n',
       'int main(){return matmul2({{1,2},{3,4}},{{5,6},{7,8}})==std::vector<std::vector<int>>({{19,22},{43,50}})?0:1;}', 'expert'),
      ('is_prime_cpp','bool is_prime_cpp(int n) {\n',
       'int main(){return is_prime_cpp(2)&&is_prime_cpp(29)&&!is_prime_cpp(49)&&!is_prime_cpp(0)?0:1;}', 'intermediate'),
    ]
    for name,prompt,harness,difficulty in defs_cpp:
        cpp.append({'id':'challenge_'+name,'category':name,'difficulty':difficulty,
                    'prompt':prompt,'test_harness':harness})
    return py,cpp


def reference_bank():
    templates={
      'python': [
        'In a Python implementation of {concept}, the function first checks the input shape and handles the empty input. An earlier version returned before completing a loop, so it processed only the first element. The corrected implementation accumulates intermediate results, then returns only after every item has been examined. A regression test specifically checks a list containing several different values.',
        'The program uses a dictionary to track {concept}. The keys are identifiers and the values are independent lists. When a new key is inserted, the program creates a fresh empty list instead of reusing a mutable default. This prevents changes made for one identifier from affecting another. Separate tests cover empty input, repeated keys, and unexpected values.'
      ],
      'cpp': [
        'A C++ implementation of {concept} accepts two vectors as const references. The function verifies compatible sizes before performing arithmetic and throws an appropriate exception for invalid dimensions. An earlier bug returned from inside the accumulation loop. The corrected function updates a local accumulator for every element pair and returns after the loop.',
        'A C++ unit test for {concept} distinguishes compilation from correctness. The generated function may compile while mishandling empty vectors or negative numbers. The test harness therefore includes a normal case, a boundary case, and an invalid-input case. The function must preserve the original interface and avoid introducing a second definition.'
      ],
      'mathematics': [
        'To analyze {concept}, begin with the definitions and state the assumptions precisely. A tempting shortcut substitutes a numerical example for a proof, but one example does not establish a universal statement. The argument next separates the necessary condition from the sufficient condition, identifies where each is used, and checks a counterexample before drawing the conclusion.',
        'For the integer operation {concept}, the intermediate computation should be checked before the final value is reported. Modular reduction preserves the remainder class but does not preserve ordinary integer magnitude. Multiplication and addition can be reduced modulo the same modulus; division requires additional conditions. These distinctions matter when constructing a correct algorithm.'
      ],
      'science': [
        'An analysis of {concept} begins by separating observation from proposed explanation. A measurement may be repeatable while the causal mechanism remains uncertain. The experiment therefore records the controlled variables, repeats the measurement, quantifies variation, and compares predictions from competing mechanisms. A correlation alone does not establish causation.',
        'The scientific report on {concept} distinguishes the conservation law from the constitutive assumption. Conservation constrains the balance over a control volume, whereas a closure relation depends on a selected physical model. The author first verifies dimensions and boundary conditions, then tests whether the limiting cases agree with known results.'
      ],
      'linux_git': [
        'A developer diagnosing {concept} first inspects the current state rather than immediately changing files. The incident log records the command, its exit code, and the relevant environment. For a shared Git repository, a reversible commit is preferable to silently rewriting a published branch. The team checks the diff and preserves a clear recovery path.',
        'A Linux workflow involving {concept} distinguishes filesystem permissions from missing paths. The technician inspects the process identity and the file owner before changing access settings. An isolated virtual environment affects installed Python packages but does not automatically modify operating-system file permissions. The final report lists each diagnostic step in order.'
      ],
      'technical': [
        'The numerical solver for {concept} separates discretization error from nonlinear-solver error. Refining the mesh can reduce spatial error without fixing an inconsistent boundary condition or an incorrect residual Jacobian. The verification plan compares residual norms, time-step convergence, and a manufactured solution. Each test isolates one source of uncertainty.',
        'In a performance study of {concept}, peak throughput alone is not sufficient evidence of efficiency. The profiler also records memory traffic, kernel launch overhead, and end-to-end latency. A faster individual kernel can leave total runtime unchanged if another operation dominates. The final comparison uses matched inputs and reports uncertainty in timings.'
      ],
      'storytelling': [
        'Mira carried a folded chart while investigating {concept}. She passed a silent bridge and reached a storage building before sunset. Earlier, she had promised to return the chart to its owner, so she did not leave it with the guard. When the door opened, she inspected the room and noticed a second copy of the same map. The discovery complicated her original plan without changing her goal.',
        'A detective entered the railway station looking for evidence about {concept}. The platform was empty and the ticket window had been closed for years. A message taped behind the counter contradicted an earlier witness statement. Rather than assuming the witness had lied, the detective checked the date and discovered that the station records had been updated later. The investigation continued from that verified fact.'
      ],
      'general_knowledge': [
        'A museum exhibit describes {concept} by giving a definition, an everyday example, and a common misconception. The first example shows how the concept applies in ordinary life. The misconception confuses two similar terms that have different meanings. The corrected explanation compares the terms and explains when each is appropriate.',
        'An introductory lesson on {concept} distinguishes reliable evidence from speculation. The instructor gives a straightforward observation and then a hypothetical exception. Students learn that an explanation must remain consistent with the conditions that have actually been established. A short summary restates the conclusion without adding unsupported details.'
      ],
    }
    concepts={
      'python':['prefix sums','binary search','prime testing','graph traversal','matrix multiplication','parsing','hash tables','recursion'],
      'cpp':['vector reduction','pointer ownership','exception safety','sorting','tree traversal','matrix products','type conversion','template deduction'],
      'mathematics':['modular arithmetic','induction','factorization','matrix rank','polynomial evaluation','combinatorics','prime divisibility','number sequences'],
      'science':['pipe friction','star formation','heat transfer','diffusion','fluid pressure','chemical reaction','measurement uncertainty','turbulence'],
      'linux_git':['branch management','commit history','file permissions','shell quoting','environment variables','recursive search','process inspection','dependency management'],
      'technical':['finite elements','CFD residuals','preconditioners','GPU memory use','attention inference','stability constraints','time integration','Jacobian assembly'],
      'storytelling':['a missing ledger','an unusual timetable','a lost compass','the old observatory','a forgotten city','the village bridge','a coded letter','the lighthouse'],
      'general_knowledge':['local government','time zones','scientific notation','electricity safety','maps','agriculture','the water cycle','public transit'],
    }
    refs=[]
    for category, items in concepts.items():
        for i, concept in enumerate(items):
            paragraphs=templates[category]
            # 3-4 paragraphs: enough to expose progression without approaching
            # the native 4096-token limit for regular reference analysis.
            text='\n\n'.join(p.format(concept=concept) for p in (paragraphs if i%2 else paragraphs[::-1]))
            text+='\n\nThe conclusion emphasizes the difference between a plausible first guess and a result verified against all stated conditions.'
            refs.append({'id':f'challenge_{category}_{i:02d}', 'category':category,
                         'difficulty':'expert' if i%2 else 'hard', 'text':text})
    return refs


def retrieval_bank():
    rows=[]
    for j, name in enumerate(['Orion','Cedar','Lotus','Quartz','Harbor','Falcon','Maple','Nimbus']):
        rows.append({'id':f'archive_{j:02d}', 'category':'single_fact',
                     'project':name, 'answer':['red','blue','green','white',
                                              'black','orange','yellow','gold'][j]})
    return rows


def main():
    R.seed(97122026)
    write('math_ladders_0097.json','math',math_bank(),
          '30 scenarios, 6 steps each: exact arithmetic through expert modular and large-integer operations.')
    write('logic_ladders_0097.json','logic',logic_bank(),
          '30 scenarios, 5 steps each: from direct order to six swaps and a reversal.')
    write('narrative_ladders_0097.json','narrative',narrative_bank(),
          '36 temporal state/belief/container scenarios, 6 explicit independently scored probes each.')
    write('terminal_advanced_0097.json','terminal',terminal_bank(),
          'Static candidate choice and safe literal command-form checks only; never execute commands.')
    py,cpp=code_banks()
    write('python_advanced_0097.json','python',py,
          'Progressive code-generation tasks with multiple deterministic unit tests in isolated sandbox.')
    write('cpp_advanced_0097.json','cpp',cpp,
          'C++17 completion, compilation and unit-test challenges in isolated sandbox.')
    (ROOT/'reference_texts_advanced_0097.json').write_text(json.dumps(
        {'version':1,'references':reference_bank()},indent=2,ensure_ascii=False)+'\n')
    write('long_retrieval_0097.json','long_retrieval',retrieval_bank(),
          'Synthetic fact retrieval at exact distances 1000/4000/16000 tokens; real historical-memory inference.')
    print('Generated 0097 fixed benchmark banks')


if __name__=='__main__':
    main()
