import subprocess
import time
import numpy as np

all_inputs_and_solutions = {
  4096: [
    ('09-nbody/files/initialized_4096', '09-nbody/files/solution_4096'),
  ],
  65536: [
    ('09-nbody/files/initialized_65536', '09-nbody/files/solution_65536'),
    # ('/dli/assessment/test_files/initialized_65536_B', '/dli/assessment/test_files/solution_65536_B'),
    # ('/dli/assessment/test_files/initialized_65536_C', '/dli/assessment/test_files/solution_65536_C')
  ]
}

expected_runtimes = {
  'V100': {
    '11': .80,
    '15': .85,
  },
  'T4': {
    '11': .90,
    '15': 1.3,
  }
}

def get_gpu_type():
  p = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
  return 'V100' if 'V100' in p.stdout else 'T4'

def compare_files(a, b):
  # Floating point calculations between the CPU and GPU vary, so in order to 
  # be able to assess both CPU-only and GPU code we compare the floating point
  # arrays within a tolerance of less than 1% of values differ by 1 or more.

  file_a = np.fromfile(a, dtype=np.float32)
  file_b = np.fromfile(b, dtype=np.float32)

  c = np.abs(file_a - file_b)
  d = c[np.where(c > 1)]

  return (len(d)/len(file_a)) < .01

def passes_with_n_of(n):
  gpu_type = get_gpu_type()
  student_output = '09-nbody/files/'
  nbodies = 2<<int(n)
  inputs_and_solutions = all_inputs_and_solutions[nbodies]

  print('Running nbody simulator with {} bodies'.format(nbodies))
  print('----------------------------------------\n')

  expected_runtime = expected_runtimes[gpu_type][n]
  print('Application should run faster than {}s'.format(expected_runtime))

  for input, solution in inputs_and_solutions:
    start = time.perf_counter()

    try:
      p = subprocess.run(['nbody', n, input, student_output], capture_output=True, text=True, timeout=5)
      ops_per_second = p.stdout
    except:
      print('Your application has taken over 5 seconds to run and is not fast enough')
      return False

    end = time.perf_counter()
  
    actual_runtime = end-start

    print('Your application ran in: {:.4f}s'.format(actual_runtime))
    if actual_runtime > expected_runtime:
      print('Your application is not yet fast enough')
      return False

    print('Your application reports ', ops_per_second)

    correct = compare_files(solution, student_output)
    if correct:
      print('Your results are correct\n')
    else:
      print('Your results are not correct\n')
      return False

  return True


def run_assessment():
  if passes_with_n_of('11') is False:
    return
  if passes_with_n_of('15') is False:
    return
  # open('/dli/assessment_results/PASSED', 'w+')
  print('Congratulations! You passed the assessment!\nSee instructions below to generate a certificate, and see if you can accelerate the simulator even more!')
