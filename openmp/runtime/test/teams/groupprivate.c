// RUN: %libomp-compile-and-run
// UNSUPPORTED: icc, gcc

// Exercises the host runtime support for the OpenMP `groupprivate` directive
// (__kmpc_groupprivate). A groupprivate variable is replicated per contention
// group: each team gets its own copy, and all threads within a team observe
// that same copy. The compiler front-ends are not required to lower the
// directive on the host, so this test drives the runtime entry point directly.
//
// A copy lives only for the lifetime of its contention group, so every check
// that reads a copy is performed inside the region; afterwards only plain copy
// address values (captured while the copies were live) are compared. A second,
// sequential teams region checks that a fresh contention group gets a
// re-initialized copy rather than stale data from the first region.

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <omp.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef int kmp_int32;
typedef struct ident {
  kmp_int32 reserved_1;
  kmp_int32 flags;
  kmp_int32 reserved_2;
  kmp_int32 reserved_3;
  char const *psource;
} ident_t;
extern int __kmpc_global_thread_num(ident_t *);
extern void *__kmpc_groupprivate(ident_t *, kmp_int32, void *, size_t);
#ifdef __cplusplus
}
#endif

#define MAX_TEAMS 8

// The original groupprivate global. Its value is used to initialize each
// contention group's copy on first access.
int gvar = 111;

int main(void) {
  uintptr_t addr[MAX_TEAMS]; // copy address captured while live (distinctness)
  int got_copy[MAX_TEAMS];   // team obtained a copy
  int init_ok[MAX_TEAMS];    // copy distinct from gvar and value-initialized
  int tag_ok[MAX_TEAMS];     // a write to the copy is observed within the region
  int same_copy_errors = 0;
  int nteams = 0;
  for (int i = 0; i < MAX_TEAMS; ++i) {
    addr[i] = 0;
    got_copy[i] = 0;
    init_ok[i] = 1;
    tag_ok[i] = 1;
  }

#pragma omp teams num_teams(MAX_TEAMS)
  {
    int team = omp_get_team_num();
    int gtid = __kmpc_global_thread_num(NULL);
    int *copy = (int *)__kmpc_groupprivate(NULL, gtid, &gvar, sizeof(gvar));

    if (team == 0)
      nteams = omp_get_num_teams();

    if (team < MAX_TEAMS) {
      got_copy[team] = 1;
      addr[team] = (uintptr_t)copy;
      // The copy must be distinct from the original global and be
      // value-initialized from it on first access.
      if (copy == &gvar || *copy != gvar)
        init_ok[team] = 0;
      // Tag the copy with the team id and read it back within the region.
      *copy = 1000 + team;
      if (*copy != 1000 + team)
        tag_ok[team] = 0;
    }

    // Every thread in the team belongs to the same contention group and must
    // therefore observe the very same copy.
#pragma omp parallel num_threads(4)
    {
      int g2 = __kmpc_global_thread_num(NULL);
      int *c2 = (int *)__kmpc_groupprivate(NULL, g2, &gvar, sizeof(gvar));
      if (c2 != copy) {
#pragma omp atomic
        ++same_copy_errors;
      }
    }
  }

  // A second, sequential teams region forms a fresh set of contention groups.
  // Each copy must be re-initialized from gvar, not retain the stale value
  // written in the first region (the contention-group root threads are very
  // likely recycled between the two regions).
  int reinit[MAX_TEAMS];
  for (int i = 0; i < MAX_TEAMS; ++i)
    reinit[i] = -1;
#pragma omp teams num_teams(MAX_TEAMS)
  {
    int team = omp_get_team_num();
    int gtid = __kmpc_global_thread_num(NULL);
    int *copy = (int *)__kmpc_groupprivate(NULL, gtid, &gvar, sizeof(gvar));
    if (team < MAX_TEAMS)
      reinit[team] = *copy;
  }

  int errors = 0;

  if (nteams <= 0 || nteams > MAX_TEAMS) {
    fprintf(stderr, "error: unexpected number of teams: %d\n", nteams);
    return 1;
  }

  if (same_copy_errors != 0) {
    fprintf(stderr, "error: %d threads saw a different copy than their team\n",
            same_copy_errors);
    ++errors;
  }

  // Each team's copy must have been value-initialized from gvar, retain this
  // team's tag, and be distinct from every other team's copy. The copies are
  // freed with their contention groups, so only the captured address values
  // (distinct while all teams were concurrently live) are compared here.
  for (int i = 0; i < nteams; ++i) {
    if (!got_copy[i]) {
      fprintf(stderr, "error: team %d never obtained a copy\n", i);
      ++errors;
      continue;
    }
    if (!init_ok[i]) {
      fprintf(stderr, "error: team %d copy not initialized from gvar\n", i);
      ++errors;
    }
    if (!tag_ok[i]) {
      fprintf(stderr, "error: team %d copy did not retain its written tag\n",
              i);
      ++errors;
    }
    for (int j = i + 1; j < nteams; ++j) {
      if (addr[i] == addr[j]) {
        fprintf(stderr, "error: teams %d and %d share a copy (%#lx)\n", i, j,
                (unsigned long)addr[i]);
        ++errors;
      }
    }
  }

  // The second region's copies must have been re-initialized from gvar.
  for (int i = 0; i < nteams; ++i) {
    if (reinit[i] != gvar) {
      fprintf(stderr,
              "error: team %d in second region saw stale value %d, expected "
              "%d\n",
              i, reinit[i], gvar);
      ++errors;
    }
  }

  if (errors == 0)
    printf("passed\n");
  return errors != 0;
}
