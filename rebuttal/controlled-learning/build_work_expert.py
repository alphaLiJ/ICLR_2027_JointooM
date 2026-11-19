"""Isolated LaCAM first-solution library with a fixed search-iteration budget.

Original upstream files and the wall-clock-budgeted library remain unchanged.
"""
import subprocess
import runtime


def main():
    root = runtime.legacy.LACAM_SOURCE
    target = runtime.ART / 'native/fixed-work'
    target.mkdir(parents=True, exist_ok=True)
    source = (root / 'lacam3/src/planner.cpp').read_text()
    source = source.replace('#include "../include/planner.hpp"', '#include "planner.hpp"')
    source = ('int jointloom_lacam_iteration_budget = 0;\n'
              'int jointloom_lacam_last_iterations = 0;\n' + source)
    needle = 'while (!OPEN.empty() && !is_expired(deadline)) {'
    assert source.count(needle) == 1
    source = source.replace(needle, 'while (!OPEN.empty() && !is_expired(deadline) &&\n'
        '         (jointloom_lacam_iteration_budget <= 0 ||\n'
        '          search_iter < jointloom_lacam_iteration_budget)) {')
    needle = 'return solution;'
    assert source.count(needle) == 1
    source = source.replace(needle, 'jointloom_lacam_last_iterations = search_iter;\n  return solution;')
    (target / 'planner_work.cpp').write_text(source)
    wrapper = (root / 'main.cpp').read_text()
    wrapper = wrapper.replace('const bool flg_no_star = false;', 'const bool flg_no_star = true;')
    wrapper = wrapper.replace('make_log(ins, solution, "lacam_log.txt", comp_time_ms, "tmp_map", seed,\n            log_short);', '')
    wrapper = wrapper.replace('#include <lacam.hpp>', '#include <lacam.hpp>\n'
        'extern int jointloom_lacam_iteration_budget;\n'
        'extern int jointloom_lacam_last_iterations;\n'
        'extern "C" int last_lacam_iterations() { return jointloom_lacam_last_iterations; }')
    wrapper = wrapper.replace('const auto solution = solve(ins, verbose - 1, &deadline, seed);',
        'jointloom_lacam_iteration_budget = static_cast<int>(time_limit_sec);\n'
        '  jointloom_lacam_last_iterations = 0;\n'
        '  const auto solution = solve(ins, verbose - 1, nullptr, seed);')
    (target / 'main_work.cpp').write_text(wrapper)
    include = '-I' + str(root / 'lacam3/include')
    commands = [
        ['g++', '-O3', '-std=c++17', '-fPIC', include, '-c',
         str(target/'planner_work.cpp'), '-o', str(target/'planner_work.o')],
        ['g++', '-O3', '-std=c++17', '-shared', '-fPIC', include,
         str(target/'main_work.cpp'), str(target/'planner_work.o'),
         str(root/'lacam3/liblacam3.a'), '-lpthread', '-o', str(target/'liblacam_work.so')],
    ]
    logs = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        logs.append(result.stdout + result.stderr)
        (target/'build.log').write_text('\n'.join(logs))
        result.check_returncode()
    runtime.json_write(target/'configuration.json', dict(
        mode='fixed search iterations; first feasible solution; no wall-clock cutoff',
        solver_seed=0, pibt_threads=10, expert_workers=4,
        source=str(root), commands=commands,
        changes=['FLG_STAR=False', 'search_iter upper bound', 'null planner deadline',
                 'no cwd-relative log', 'iteration count accessor']))
    print(target/'liblacam_work.so', flush=True)


if __name__ == '__main__':
    main()
