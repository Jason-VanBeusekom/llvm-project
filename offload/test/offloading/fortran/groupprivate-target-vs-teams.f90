! Offloading execution test for the OpenMP `groupprivate` directive.
!
! A groupprivate variable gets one copy per contention group. A plain `target`
! region establishes a single initial team (one contention group), so it must
! behave the same as `target teams num_teams(1)`. This checks that groupprivate
! storage works on the device in both forms and yields identical results.

! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=60
! RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic

module gp_mod
  implicit none
  integer, save :: gv
  !$omp groupprivate(gv)
end module gp_mod

program main
  use gp_mod
  implicit none
  integer :: r1, r2

  ! Plain target: the initial team is the contention group.
  r1 = -1
  !$omp target map(from: r1)
    gv = 21
    r1 = gv * 2
  !$omp end target

  ! target teams num_teams(1): a single team, i.e. one contention group.
  r2 = -1
  !$omp target teams num_teams(1) map(from: r2)
    gv = 21
    r2 = gv * 2
  !$omp end target teams

  print '(A, I0)', "r1=", r1
  print '(A, I0)', "r2=", r2
end program main

! CHECK: r1=42
! CHECK: r2=42
