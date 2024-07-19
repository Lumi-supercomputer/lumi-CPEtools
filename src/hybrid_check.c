//
// Demo program part of the UAntwerp VSC tutorials and LUMI consortium LUMI
// tutorials (and production tools).
//
// This program is used to demonstrate starting a hybrid OpenMP/MPI-program.
// It is essentially a hello world program, but with an added element that does
// make the source code more difficult: To make it easy to check that all processes
// and threads run where we expect it, we make sure that the output is ordered
// according to MPI rank and thread number.
//

#define _GNU_SOURCE   // Needed to get gethostname from unistd.h and sched_getcpu from sched.h
//#define HAVE_NUMALIB 1

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <sched.h>
#ifdef HAVE_NUMALIB
#include <numa.h>
#endif
#include <libgen.h>

#ifdef WITH_MPI
#include <mpi.h>
#endif
#ifdef WITH_OMP
#include <omp.h>
#endif

#define max(a,b)  ( (a) < (b) ? (b) : (a) )
#define min(a,b)  ( (a) > (b) ? (b) : (a) )
#define IWIDTH(v) ( (v) > 9 ?  ((int)(floor(log10((double)(v)))) + 1 ) : (1) )


#define EXIT_SUCCESS         0
#define EXIT_WRONG_ARGUMENT  1

#define HOSTNAMELENGTH       20         // Not really good practice to fix the length of strings as it can cause
                                        // buffer overflows, but we've been careful to ensure that this does not
                                        // occur and have no problems with chopping of part of the hostname.
#define LABELLENGTH          20
#define MAXLINE_CPUINFO      1000       // Line length in /proc/cpuinfo. It doesn't matter if some lines are longer,
                                        // the read is protected
#define SAMPLE_BLOCK         10000L

typedef struct {
    long      corenum;                  // Number of the core on which we are running. Long for 64-bit alignment on 64-bit systems.
    cpu_set_t mask;                     // Mask as returned by sched_getaffinity.
    short     numamask[__CPU_SETSIZE];  // For each logical core: -1 if not in the mask, SHRT_MIN if not corresponding to a processor,
                                        // Number of the NUMA node when in the mask.
} t_threadData;

typedef struct {
	t_threadData *threadData;                // Will point to an array containing the OS core number for each OpenMP thread
                                             // We'll do a dirty trick and actually start that array at the firstthread field
                                             // when we allocate memory.
    int          mpi_myrank;                 // MPI rank of the current process
    int          openmp_numthreads;          // Number of threads in the current MPI process
    int          ncpus;                      // Number of CPUs on the node as detected by Linux.
    int          slurm_procid;               // Slurm global rank from SLURM_PROCID, or -1
    int          slurm_nodeid;               // Slurm relative node ID from SLURM_NODEID, or -1
    int          slurm_localid;              // Slurm local rank from SLURM_LOCALID, or -1
    char         hostname[HOSTNAMELENGTH+1]; // Name of the host on which this process is running
    char         label[LABELLENGTH+1];       // Label (if specified)
    t_threadData firstthread;                // To make sure that the end of the struct is properly aligned for integers.
} t_rankData;

typedef struct {
	int mpi_myrank;                          // My MPI rank
	int mpi_numranks;                        // Total number of MPI ranks
    int openmp_numthreads;                   // Number of OpenMP threads.
    int max_openmp_numthreads;               // Maximal number of threads per process over all MPI processes.
    int min_openmp_numthreads;               // Minimal number of threads per process over all MPI processes.
    int sum_openmp_numthreads;               // Total number of OpenMP threads over all MPI processes.
} t_configInfo;

//******************************************************************************
//
// print_help()
//
// Prints help information.
//

void print_help( const char *exe_name ) {

	fprintf( stderr,
        "%s\n"
		"\n",
        exe_name
	);
#if defined( WITH_OMP ) && defined ( WITH_MPI )
	fprintf( stderr,
		"This program prints out useful information on the distribution of processes\n"
		"and threads in a hybrid MPI/OpenMP application.\n"
		"It also provides information on where each thread is running and how threads\n"
		"are bound to CPUs. It also supports MPMD startup, though you would use the\n"
		"same executable for each part of the heterogeneous job and instead use a\n"
		"different label for each part.\n"
	);
#elif defined( WITH_OMP )
	fprintf( stderr,
		"This program prints out useful information on the distribution of threads\n"
		"in an OpenMP application.\n"
		"It also provides information on where each thread is running and how threads\n"
		"are bound to CPUs.\n"
	);
#elif defined( WITH_MPI )
	fprintf( stderr,
		"This program prints out useful information on the distribution of processes\n"
		"in an MPI application.\n"
		"It also provides information on where each process is running and how it\n"
		"is bound to CPUs. It also supports MPMD startup, though you would use the\n"
		"same executable for each part of the heterogeneous job and instead use a\n"
		"different label for each part.\n"
	);
#else
	fprintf( stderr,
		"This program prints out useful information on the core on which a serial\n"
		"process is running.\n"
	);
#endif
	fprintf( stderr,
		"\n"
		"Flags accepted:\n"
		" -h             produce help information and exit\n"
		" -l <label>     specify a label to use in the output for this process\n"
		"                This can be used to simulate using different executable names\n"
		"                in some scenarios.\n"
		" -w <time>      keeps all allocated CPUs busy for approximately <time> seconds\n"
		"                computing the surface of the Mandelbrot fractal with a naive\n"
		"                Monte Carlo algorithm so that a user can logon to the nodes\n"
		"                and see what is happening. At the end it will print the\n"
		"                allocation of the threads again as the OS may have improved\n"
		"                the spreading of the threads over the cores.\n"
	    "                If different values are given for different parts in a\n"
		"                heterogeneous job, the largest will be used and applied to all\n"
        "                components of the heterogeneous job.\n"
		" -n             Show the NUMA affinity mask: Once ASCII character per virtual\n"
		"                core, where a number or capital letter denotes that the core\n"
		"                can be used and the number or letter denotes the NUMA group\n"
		"                (or a * if the number of the NUMA group would be 36 or larger)\n"
		"                and a dot denotes that the core is not used.\n"
		" -r             Numeric representation of the affinity mask as a series of\n"
		"                ranges of cores.\n"
		" --no-hostname  Do not show the hostname in the output.\n"
		" --no-cpu       Do not show the CPU number in the output.\n"
#if defined( WITH_MPI )
		" --slurmvars    Slow the value of SLURM_NODEID, SLURM_LOCALID and SLURM_PROCID\n"
		"                for each rank. This can be useful to study Cray MPICH rank\n"
		"                reordering.\n"
#endif
#if defined( WITH_MPI )
		" --fp           Format showing host, cpu and numeric mask, but no Slurm\n"
		"                variables.\n"
#else
    	" --fp           Format showing host, cpu and numeric mask.\n"
#endif
#if defined( WITH_MPI )
		" --fr           Format showing slurm variables and numeric mask, but no hostname\n"
		"                or cpu number, making the output fully reproducible when running\n"
		"                on exclusive nodes.\n"
#else
		" --fr           Format showing numeric mask, but no hostname or cpu number,\n"
		"                making the output fully reproducible when running on exclusive\n"
		"                nodes.\n"
#endif
		"\n"
	);
#if defined( WITH_MPI )
	fprintf( stderr,
		"In a heterogeneous job, the -w, -n and -r options need to be specified for\n"
        "only one of the components of the job but will apply to all components.\n"
        "\n"
    );
#endif
	fflush( stderr );

}



//******************************************************************************
//
// get_numcpus( )
//
// Get the number of CPUs as seen by Linux in /proc/cpuinfo
//
int get_numcpus() {

  FILE *fp;
  char line[MAXLINE_CPUINFO];
  char *c;
  int  ncpu = 0;

  if ( (fp = fopen( "/proc/cpuinfo", "r" )) != NULL ) {

	// Count the number of lines that start with processor.
    while( fgets( line, MAXLINE_CPUINFO, fp ) != NULL )
      if ( strncmp( line, "processor", 9 ) == 0 ) ncpu++;

	  fclose( fp );

  }

  return ncpu;

}



//******************************************************************************
//
// get_slurmvar( const char *env_var )
//
// Get the numeric value of the Slurm environment variable, or -1 if the
// variable is not set.
//
int get_slurmvar( const char *env_var  ) {

    char *env_value = getenv( env_var );

    if ( env_value == (char *) NULL )
    	return -1;
    else {
        return atoi( env_value );
    }

}



//******************************************************************************
//
// short get_cpu_to_numanode( short *cpu_to_numanode, int ncpus ) )
//
// Build the CPU-to-NUMA node map.
//
// Input arguments:
//   * short *cpu_to_numanode: Points to the first element of an array of at
//     least ncpus shorts.
//   * int ncpus: Number of CPUs in the node.
//
// Return value: The number of NUMA nodes on the node (as an integer)
//
#ifdef HAVE_NUMALIB

int get_cpu_to_numanode( short * const cpu_to_numanode, const int ncpus ) {

	int numanode;
	int maxnode = 0;

	if ( numa_available() == -1 ) {
		// No NUMA, set all entries to 0
		for ( int cpu = 0; cpu < ncpus; cpu++ ) cpu_to_numanode[cpu] = 0;
		return 1;
	}

	for ( int cpu = 0; cpu < ncpus; cpu++ ) {
		numanode = (short) numa_node_of_cpu( cpu );
		maxnode = ( ( numanode > maxnode ) ? numanode : maxnode );
		cpu_to_numanode[cpu] = (short) numanode;
	}

	return maxnode + 1;

}

#else

// We need to use a kludge implementation reading some Linux files

int get_cpu_to_numanode( short * const cpu_to_numanode, const int ncpus ) {

	int current_node;
	const int numfields = (ncpus + 31) / 32;  // Number of fields in the files with the masks.

	uint32_t masks_read[(__CPU_SETSIZE+31)/32];

#ifdef DEBUG
	printf( "get_cpu_to_numanode: Check whether the /sys/devices/system/node/node*/cpumap files have %d fields\n", numfields ); fflush( stdout );
#endif

	// Clear the CPU to node list.
    for ( int cpu = 0; cpu < ncpus; cpu++ ) cpu_to_numanode[cpu] = -1;

    // Now look in a number of files with the CPUs for each NUMA node
    current_node = 0;

    do {

    	int cpu;
    	char fname[45];
        FILE *fp;

        sprintf( fname, "/sys/devices/system/node/node%d/cpumap", current_node );

        if ( ( fp = fopen( fname, "r" ) ) == NULL ) break;   // No more NUMA nodes

        // Format of the file: 00000000,00000000,00000000,ffff0000,00000000,00000000,00000000,ffff0000
        for ( int t1 = numfields -  1; t1 >= 0; t1-- ) {
        	if ( fscanf( fp, "%x", masks_read + t1 ) != 1 ) {
        		fprintf( stderr, "Failed to process %s\n", fname );
        		exit( 1 );
        	}
        	if ( t1 > 0 ) fgetc( fp );  // Skip over the comma.
        }  // End for-loop for reading the cpumap file.

        fclose( fp );

        // Now process the information just read.
        cpu = 0;
        for ( int t1 = 0; t1 < numfields; t1++ ) {
        	for ( int t2 = 0; t2 < min( 32, ncpus - 32*t1 ); t2++ ) {
        		// At this point cpu = 32*t1 + t2.
                if ( masks_read[t1] & (unsigned int) 1 ) cpu_to_numanode[cpu] = current_node;
                masks_read[t1] >>= 1;
        		cpu++;
        	}
        }

        current_node++;

    }
    while ( 1 ); // We get out of the loop with a break statement when no files can be read anymore.
                 // Not the cleanest coding style though.

    return current_node;  // This is also the number of nodes read.

}

#endif


//******************************************************************************
//
// mask_to_numanode( t_threadData *threadData, int ncpus, short *cpu_to_numanode )
//
// Convert the mask (field mask) to a numamask array (field numamask) where
// each entry is -1 if not part of the mast, SHRT_MIN if not even a valid
// logical CPU on the node and the number of the numa node otherwise
//
void mask_to_numanode( t_threadData *threadData, const int ncpus, const short * const cpu_to_numanode ) {

	for ( int t1 = 0; t1 < ( ncpus < __CPU_SETSIZE ? ncpus : __CPU_SETSIZE ); t1++ )
		if ( CPU_ISSET( t1, &threadData->mask ) )
			threadData->numamask[t1] = cpu_to_numanode[t1];
		else
            threadData->numamask[t1] = -1;

	for ( int t1 = ( ncpus < __CPU_SETSIZE ? ncpus : __CPU_SETSIZE ); t1 < __CPU_SETSIZE; t1++ )
		threadData->numamask[t1] = SHRT_MIN;

}


//******************************************************************************
//
// numamask_to_ASCII( char *ascii_mask, short *numamask, int ncpus )
//
// Input arguments:
//  * char * ascii_mask: pointer to a string area sufficiently large to store
//    ncpus + 1 characters
//  * short *numamask: Pointer to the mask in NUMAMASK format
//  * int ncpus: The number of cpus
//

void numamask_to_ASCII( char * const ascii_mask, const short * const numamask, const int ncpus ) {

	for ( int cpu = 0; cpu < ncpus; cpu++ )
		if      ( numamask[cpu] < 0 )  ascii_mask[cpu] = '.';
		else if ( numamask[cpu] < 10 ) ascii_mask[cpu] = '0' + numamask[cpu];
		else if ( numamask[cpu] < 36 ) ascii_mask[cpu] = 'A' + (numamask[cpu] - 10);
		else                           ascii_mask[cpu] = '*';

	ascii_mask[ncpus] = '\0';

}


//******************************************************************************
//
// numamask_to_range( char *rangemask, short *numamask, int ncpus )
//
// Input arguments:
//  * char * rangemask: pointer to a string area sufficiently large to store
//    ncpus + 1 characters
//  * short *numamask: Pointer to the mask in NUMAMASK format
//  * int ncpus: The number of cpus
//

void numamask_to_range( char * const rangemask, const short * const numamask, const int ncpus ) {

	int start = -1;
	int output = 0;  // 0 as long as we have not written anything to rangemask.
	char number[2*IWIDTH(__CPU_SETSIZE)+2];  // Space enough for number-number.
    const int loop_limit = min( ncpus, __CPU_SETSIZE );

	rangemask[0] = '\0';

	for ( int cpu = 0; cpu <= loop_limit; cpu++ ) { // One iteration more than the number of CPUs to close.

		if ( ( cpu == loop_limit ) || ( numamask[cpu] < 0 ) ) {

			if ( start >= 0 ) { // We've isolated a new range ending at the previous cpu.

				if ( output ) strcat( rangemask, ", " );
				else          output = 1;

				if ( start == (cpu - 1) ) {
					// Just a single core in the range
                    sprintf( number, "%d", start );
				} else {
					// True range from start to cpu-1
					sprintf( number, "%d-%d", start, cpu - 1 );
				}

				strcat( rangemask, number );
				start = -1;

			}

		} else {

			if ( start < 0 ) start = cpu;

		}

	}

}


//******************************************************************************
//
// get_args( argc, argv, mpi_myrank, *option_label, *wait_time,
//           *show_numamask, *show_rangemask )
//
// Gets the input arguments.
//
// Arguments:
//  * argc: Argument count from the main function
//  * argv: Argument values from the main function
//  * mpi_myrank: MPI rank to ensure that help is printed only once.
//    Since in a heterogeneous program some program arguments may only
//    be given for the second instance, we cannot avoid that other error
//    messages will be printed once for each MPI rank in the job component.
//  * option_label: Space of at least LABELLENGTH+1 bytes to receive the
//    (possibly truncated) label from the -l argument. Empty string if no
//    label is specified.
//  * wait_time: Argument of the -w flag, or 0.
//  * show_numamask: On return nonzero if -a is specified, zero otherwise.
//  * show_rangemask: On return nonzero if -n is specified, zero otherwise.
//

void get_args( int argc, char **argv, int mpi_myrank,
		       char *option_label, long *wait_time,
			   unsigned int *show_numamask, unsigned int *show_rangemask,
			   unsigned int *show_hostname, unsigned int *show_cpu,
			   unsigned int *show_slurmvars ) {

	char *exe_name;

	// Make sure we always return initialised variables, whatever happens.
	// option_label[0] = '\0';
	// Default label is the name of the executable derived from argv[0].
	strncpy( option_label, basename( argv[0] ), LABELLENGTH );
	option_label[LABELLENGTH] = '\0';  // Just to be very sure that the string is null-terminated.
	*wait_time = 0L;
	*show_numamask = 0;
	*show_rangemask = 0;
	*show_hostname = 1;
	*show_cpu = 1;
	*show_slurmvars = 0;

	// Remove the program name
	exe_name = basename( *argv++ );
	argc--;

	while ( argc-- ) {

		if ( strcmp( *argv, "-h") == 0 ) {
			if ( mpi_myrank == 0 ) print_help( exe_name );
			exit( EXIT_SUCCESS );
		} else if ( strcmp( *argv, "-n") == 0 ) {
			*show_numamask = 1;
		} else if ( strcmp( *argv, "-r") == 0 ) {
			*show_rangemask = 1;
		} else if ( strcmp( *argv, "-l") == 0 ) {
			if ( argc == 0 ) {  // No arguments left, so we have a problem.
				fprintf( stderr, "%s: No label found for -l.\n", exe_name );
				exit( EXIT_WRONG_ARGUMENT );
			}
            argv++; argc--; // Skip to the next argument which should be the label.
            strncpy( option_label, *argv, (size_t) LABELLENGTH );
            option_label[LABELLENGTH] = '\0';  // Just to be very sure that the string is null-terminated.
		} else if ( strcmp( *argv, "-w" ) == 0 ) {
			char *endptr;

			if ( argc == 0 ) {  // No arguments left so we have a problem.
				fprintf( stderr, "%s: No waiting time found for -w.\n", exe_name );
				exit( EXIT_WRONG_ARGUMENT );
			}
            argv++; argc--; // Skip to the next argument which should be the waiting time.
            *wait_time = strtol( *argv, &endptr, (int) 10 );
            if ( ( wait_time < 0 ) || ( *endptr != '\0' ) ) {
				fprintf( stderr, "%s: No valid waiting time found for -w.\n", exe_name );
				exit( EXIT_WRONG_ARGUMENT );
			}
		} else if ( strcmp( *argv, "--no-hostname" ) == 0 ) {
			*show_hostname = 0;
		} else if ( strcmp( *argv, "--no-cpu" ) == 0 ) {
			*show_cpu = 0;
#if defined( WITH_MPI )
		} else if ( strcmp( *argv, "--show-slurmvars" ) == 0 ) {
			*show_slurmvars = 1;
#endif
		} else if ( strcmp( *argv, "--fp" ) == 0 ) {
			*show_slurmvars = 0;
			*show_hostname  = 1;
			*show_cpu       = 1;
			*show_numamask  = 0;
			*show_rangemask = 1;
		} else if ( strcmp( *argv, "--fr" ) == 0 ) {
#if defined( WITH_MPI )
			*show_slurmvars = 1;
#else
			*show_slurmvars = 0;
#endif
			*show_hostname  = 0;
			*show_cpu       = 0;
			*show_numamask  = 0;
			*show_rangemask = 1;
		} else {
			fprintf( stderr, "%s: Illegal flag found: %s\n", exe_name, *argv);
			exit( EXIT_WRONG_ARGUMENT );
		}

		argv++;

	}


	return;

}


//******************************************************************************
//
// data_gather_print(  )
//
// Gather all data on where threads run and print the summary.
//
// This function is not time-critical (in fact the whole program is not), so we
// don't care recomputing stuff that has already been computed for the sake of
// a simpler, shorter argument list.
//


int data_gather_print( const t_configInfo *configInfo, const char *option_label,
		               unsigned int show_numamask, unsigned int show_rangemask,
					   unsigned int show_hostname, unsigned int show_cpu,
					   unsigned int show_slurmvars ) {

	const int mssgID = 1;

    int error;
	int label_length;
	int max_label_length = 0;                // Make sure it has a reasonable initialization on all MPI ranks
	                                         // even though we intend to only use it on rank 0.

    t_rankData *my_rankData;                 // Rank data for this MPI process
    t_rankData *buf_rankData;                // Buffer to receive data from another process.
    int my_rankData_size, buf_rankData_size;
    short cpu_to_numanode[__CPU_SETSIZE];
    int nnumanodes;

#ifdef WITH_MPI
    MPI_Request request;
    MPI_Status status;
#endif

    // Compute the maximum length of the labels, and take into account that in a heterogeneous
    // job some parts may not be labeled.
    // Only MPI rank 0 needs max_label_length.
    label_length = strlen( option_label ); // We can in fact always use strlen as option_label has been initialised to the empty string.
#ifdef WITH_MPI
    MPI_Reduce( &label_length, &max_label_length, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD );
#else
    max_label_length = label_length;
#endif

    // Create the t_rankData data structures.

    // Sending buffer: We know how many OpenMP threads we have so we can size correctly.
    my_rankData_size = sizeof( t_rankData ) + (configInfo->openmp_numthreads - 1) * sizeof( t_threadData );
    my_rankData = (t_rankData *) malloc( my_rankData_size );
    if ( my_rankData == NULL ) { fprintf( stderr, "ERROR: Memory allocation for my_rankData failed.\n" ); return 1; };
    my_rankData->threadData = &(my_rankData->firstthread);

    if ( configInfo->mpi_myrank == 0 ) {
    	// Receiving buffer: We may get messages from ranks with more threads to size for the maximum that we can get.
        buf_rankData_size = sizeof( t_rankData ) + (configInfo->max_openmp_numthreads - 1) * sizeof( t_threadData );
        buf_rankData = (t_rankData *) malloc( buf_rankData_size );
        if ( buf_rankData == NULL ) { fprintf( stderr, "ERROR: Memory allocation for buf_rankData failed.\n" ); return 2; };
        buf_rankData->threadData = &(buf_rankData->firstthread);
    }  // end if ( mpi_myid == 0 )

    //
    // Gather all data for the current MPI process and store in the my_rankData structure.
    //

    my_rankData->mpi_myrank = configInfo->mpi_myrank;
    my_rankData->openmp_numthreads = configInfo->openmp_numthreads;
    my_rankData->ncpus = get_numcpus();
    my_rankData->slurm_procid = get_slurmvar( "SLURM_PROCID" );
    my_rankData->slurm_nodeid = get_slurmvar( "SLURM_NODEID" );
    my_rankData->slurm_localid = get_slurmvar( "SLURM_LOCALID" );
    gethostname( my_rankData->hostname, (size_t) (HOSTNAMELENGTH+1) );
    strcpy( my_rankData->label, option_label );
    nnumanodes = get_cpu_to_numanode( cpu_to_numanode, my_rankData->ncpus );
#ifdef WITH_OMP
#pragma omp parallel
#endif
    {
        int openmp_myid;

        /* Get OpenMP information. */
#ifdef WITH_OMP
        openmp_myid = omp_get_thread_num();
#else
        openmp_myid = 0;
#endif
        my_rankData->threadData[openmp_myid].corenum = (long) sched_getcpu();
        if ( sched_getaffinity( 0, sizeof( cpu_set_t ), &my_rankData->threadData[openmp_myid].mask ) == -1 )
            perror( "data_gather_print" );
        mask_to_numanode( &my_rankData->threadData[openmp_myid], my_rankData->ncpus, cpu_to_numanode );
    } // End of OpenMP parallel section.

    //
    // Finally print the results.
    // We'll do all printing on the master thread of MPI rank 0 so that
    // the output is sorted as it should.
    // To avoid having to build complex MPI datatypes, we assume a homogeneous
    // cluster in the sense that all nodes used have the same CPU family.
    //

    // Send data to process with rank 0.
#if defined( DEBUG ) && defined( WITH_MPI )
    printf( "DEBUG: Rank %d: Sending a message of %d bytes, claiming %d threads, %d CPUs, host %s.\n",
    		configInfo->mpi_myrank, my_rankData_size, my_rankData->openmp_numthreads, my_rankData->ncpus, my_rankData->hostname );
    fflush( stdout );
#endif
#ifdef WITH_MPI
    error = MPI_Isend( my_rankData, my_rankData_size, MPI_BYTE, 0, mssgID, MPI_COMM_WORLD, &request );
#endif
#if defined( DEBUG ) && defined( WITH_MPI )
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    if ( configInfo->mpi_myrank == 0 ) {

    	const char *empty_string = "";
    	const char *sep_colon = ": ";

    	char numamask[__CPU_SETSIZE+1];
    	char rangemask[(IWIDTH(__CPU_SETSIZE)+2)*__CPU_SETSIZE];

        const char *label_sep = ( ( max_label_length > 0 ) ? sep_colon : empty_string );

        for ( int c1 = 0; c1 < configInfo->mpi_numranks; c1++ ) {

#ifdef WITH_MPI
            error = MPI_Recv( buf_rankData, buf_rankData_size, MPI_BYTE, c1, mssgID, MPI_COMM_WORLD, &status );
#else
            memcpy( buf_rankData, my_rankData, my_rankData_size );
#endif
            buf_rankData->threadData = &(buf_rankData->firstthread); // Because the pointer from the other process doesn't make sense.
#if defined( DEBUG ) && defined( WITH_MPI )
    printf( "DEBUG: Rank 0: Received a message of %d bytes from rank %d, claiming %d threads, %d CPUs, host %s.\n",
    		my_rankData_size, c1, buf_rankData->openmp_numthreads, buf_rankData->ncpus, buf_rankData->hostname );
    fflush( stdout );
#endif

            // The min function should not be needed, but it prevents overrunning memory in case of a corrupt
            // message so also helps to avoid segmentation violations.
            for ( int c2 = 0; c2 < min( buf_rankData->openmp_numthreads, configInfo->max_openmp_numthreads ); c2++ ) {
            	printf( "++ %-*s%s", max_label_length, buf_rankData->label, label_sep );
#if defined( WITH_MPI )
            	printf( "MPI rank %3d/%-3d ",
            			buf_rankData->mpi_myrank, configInfo->mpi_numranks );

#endif
            	if ( show_slurmvars )
            		printf( "NodeID/LocID/GlobID %2d/%3d/%3d ",
            				buf_rankData->slurm_nodeid, buf_rankData->slurm_localid, buf_rankData->slurm_procid );
            	if ( show_hostname )
            		printf( "host %s ", buf_rankData->hostname );
#if defined( WITH_OMP )
            	// OpenMP case or hybrid case
            	printf( "OpenMP thread %3d/%-3d ",
            			c2, buf_rankData->openmp_numthreads );
#endif
            	if ( show_cpu )
            	    printf( "cpu %3ld/%-3d ", buf_rankData->threadData[c2].corenum, buf_rankData->ncpus );
            	if ( show_numamask || show_rangemask ) {
            		printf( "mask " );
            		if ( show_numamask ) {
            			numamask_to_ASCII( numamask, buf_rankData->threadData[c2].numamask, buf_rankData->ncpus );
            			printf( "%s ", numamask );
            		}
                    if ( show_rangemask ) {
                    	numamask_to_range( rangemask, buf_rankData->threadData[c2].numamask, buf_rankData->ncpus );
            			printf( "(%s)", rangemask );
                    }
            	}
            	printf(  "\n" );

            }  // end for ( int c2...
        }  // end for ( int c1...

        printf( "\n" );

    } // end if ( mpi_myrank == 0 )

    // Make sure all data is send before clearing the buffer.
#ifdef WITH_MPI
    MPI_Wait( &request, &status );
#endif

    // Free memory allocated with malloc
    free( my_rankData );
    if ( configInfo->mpi_myrank == 0 ) free( buf_rankData );

    return 0;

}

typedef unsigned short t_randState[4];  // We need only three but we take 4 for alignment.

int mandelSurface( const t_configInfo * const configInfo, const long wait_time, double * const mandel_surface ) {

	clock_t start, current;
	long samples =    0L;
	long in_fractal = 0L;

	const int max_cycles = 100;

	t_randState *randState;

	// Create and initialize randState

	randState = malloc( configInfo->openmp_numthreads * sizeof( t_randState ) );
	for ( int t1 = 0 ; t1 < configInfo->openmp_numthreads; t1++ ) {
		int a = t1 + configInfo->max_openmp_numthreads * configInfo->mpi_myrank;
	    int b = 2 * t1 + 5 * configInfo->max_openmp_numthreads * configInfo->mpi_myrank;
	    int c = 3 * t1 + 15 * configInfo->max_openmp_numthreads * configInfo->mpi_myrank;
        randState[t1][0] = (unsigned short) a;
        randState[t1][1] = (unsigned short) b;
        randState[t1][2] = (unsigned short) c;
	}

	// Start synchronized
#ifdef WITH_MPI
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    // Get the start clock
    start = clock();
    current = start;

    // clock returns an estimate for the total time for all threads which is why we have to use
    // the factor configInfo->openmp_numthreads.
    while ( (current - start) <  (wait_time * CLOCKS_PER_SEC * (long) configInfo->openmp_numthreads) ) {

#ifdef WITH_OMP
#pragma omp parallel for reduction(+:samples,in_fractal)
#endif
        for ( long t1 = 0 ; t1 < (long) configInfo->openmp_numthreads * SAMPLE_BLOCK; t1++ ) {

        	 double c_r, c_i;
        	 double z_r, z_i;
        	 double z_rs, z_is;
        	 int    cycle;

#ifdef WITH_OMP
             c_r = -2.25 + 3.0 * erand48( randState[omp_get_thread_num()] );
             c_i = -0.75 + 1.5 * erand48( randState[omp_get_thread_num()] );
#else
             c_r = -2.25 + 3.0 * erand48( randState[0] );
             c_i = -0.75 + 1.5 * erand48( randState[0] );
#endif

             // First iteration starts with z=0
             z_r   = c_r;
             z_i   = c_i;
             z_rs  = z_r*z_r;
             z_is  = z_i*z_i;
             cycle = 0;

             while ( ( z_rs + z_is <= 4.0 ) && ( cycle < max_cycles ) ) {

            	 // z <- zˆ2 + c = (z_r + i*z_i)^2 + (c_r+i*c_i)
            	 //              = ((z_r^2  - z_i^2) + c_r) + i*(2*z_r*z_i + c_i)
            	 z_i  = 2 * z_r * z_i + c_i;
            	 z_r  = z_rs - z_is   + c_r;
            	 z_rs = z_r * z_r;
            	 z_is = z_i * z_i;
            	 cycle++;

             }

             if ( z_rs + z_is <= 4.0 ) in_fractal++;
             samples++;

        }  // End for ( long t1 = 0; ...)

        current = clock();

    }  // End while ( (current - start) ...

    // Now make a global sum over all MPI ranks of in_fractal and samples.

    long in[2], sum[2];
    in[0] = in_fractal;
    in[1] = samples;
#ifdef WITH_MPI
    MPI_Allreduce( in, sum, 2, MPI_LONG, MPI_SUM, MPI_COMM_WORLD );
#else
    sum[0] = in[0];
    sum[1] = in[1];
#endif

    *mandel_surface = 3.0 * 1.5 * (double) sum[0] / (double) sum[1];

    return 0;

}


/******************************************************************************
This is a simple hello world program. Each thread prints out the rank of its
MPI process and its OpenMP thread number, and the total number of MPI ranks
and OpenMP threads per process.
******************************************************************************/

int main( int argc, char *argv[] )
{
	char option_label[LABELLENGTH+1];   // Space for the label (if used).

	long wait_time, max_wait_time;
	unsigned int show_numamask, in_show_numamask;
	unsigned int show_rangemask, in_show_rangemask;
	unsigned int show_hostname, in_show_hostname;
	unsigned int show_cpu, in_show_cpu;
	unsigned int show_slurmvars, in_show_slurmvars;

	t_configInfo configInfo;

	t_rankData *my_rankData;                 // Rank data for this MPI process
    t_rankData *buf_rankData;                // Buffer to receive data from another process.
    int my_rankData_size, buf_rankData_size;
    int error;
#ifdef WITH_MPI
    MPI_Request request;
    MPI_Status status;
#endif

#ifdef WITH_MPI
    MPI_Init( &argc, &argv );            // Standard way of initializing a MPI program.
#endif

    //
    // Initializations
    //

    // Get basic info: Number of processes and the number of threads in this process
#ifdef WITH_MPI
    MPI_Comm_size( MPI_COMM_WORLD, &configInfo.mpi_numranks );
    MPI_Comm_rank( MPI_COMM_WORLD, &configInfo.mpi_myrank );
#else
    configInfo.mpi_numranks = 1;
    configInfo.mpi_myrank   = 0;
#endif

#ifdef WITH_OMP
#pragma omp parallel shared( configInfo )   // The shared clause is not strictly needed as that variable will be shared by default.
    { if ( omp_get_thread_num() == 0 ) configInfo.openmp_numthreads = omp_get_num_threads(); } // Must be in a parallel session to get the proper number.
#else
    configInfo.openmp_numthreads = 1;
#endif

    // We'll make sure that all fields of configInfo have valid values on all MPI ranks.
#ifdef WITH_MPI
    MPI_Allreduce( &configInfo.openmp_numthreads, &configInfo.max_openmp_numthreads, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );
    MPI_Allreduce( &configInfo.openmp_numthreads, &configInfo.min_openmp_numthreads, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD );
    MPI_Allreduce( &configInfo.openmp_numthreads, &configInfo.sum_openmp_numthreads, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
#else
    configInfo.max_openmp_numthreads = configInfo.openmp_numthreads;
    configInfo.min_openmp_numthreads = configInfo.openmp_numthreads;
    configInfo.sum_openmp_numthreads = configInfo.openmp_numthreads;
#endif

    //
    // Read the command line arguments
    //
    get_args( argc, argv, configInfo.mpi_myrank, option_label, &wait_time, &in_show_numamask, &in_show_rangemask,
    		  &in_show_hostname, &in_show_cpu, &in_show_slurmvars );

    // Compute the maximum wait time: In a heterogeneous situation, one may have given a different load
    // for each process set yet all tasks must be able to decide if we should put load or not. In fact.
    // In the implementation we chose to have the same length of load for every MPI rank.
#ifdef WITH_MPI
    MPI_Allreduce( &wait_time,         &max_wait_time,  1, MPI_LONG,     MPI_MAX, MPI_COMM_WORLD );
    MPI_Allreduce( &in_show_numamask,  &show_numamask,  1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD );
    MPI_Allreduce( &in_show_rangemask, &show_rangemask, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD );
    MPI_Allreduce( &in_show_hostname,  &show_hostname,  1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD );
    MPI_Allreduce( &in_show_cpu,       &show_cpu,       1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD );
    MPI_Allreduce( &in_show_slurmvars, &show_slurmvars, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD );
#else
    max_wait_time  = wait_time;
    show_numamask  = in_show_numamask;
    show_rangemask = in_show_rangemask;
    show_hostname  = in_show_hostname;
    show_cpu       = in_show_cpu;
    show_slurmvars = in_show_slurmvars;
#endif

    // Print some information on the data just computed.
    if ( configInfo.mpi_myrank == 0 ) {
    	if ( configInfo.mpi_numranks == 1 ) {
    		if ( configInfo.max_openmp_numthreads == 1 ) {
    			printf( "\nRunning a single thread in a single process.\n" );
    		} else {
    			printf( "\nRunning %d threads in a single process\n", configInfo.max_openmp_numthreads );
    		}
    	} else {
            if ( configInfo.max_openmp_numthreads == 1 ) {
            	printf( "\nRunning %d single-threaded MPI ranks.\n", configInfo.mpi_numranks );
            } else if ( configInfo.max_openmp_numthreads == configInfo.min_openmp_numthreads ) {
            	printf( "\nRunning %d MPI ranks with %d threads each (total number of threads: %d).\n",
            			configInfo.mpi_numranks, configInfo.max_openmp_numthreads, configInfo.sum_openmp_numthreads );
            } else {
            	printf( "\nRunning %d MPI ranks with between %d and %d threads each (total number of threads: %d).\n",
            			configInfo.mpi_numranks, configInfo.min_openmp_numthreads, configInfo.max_openmp_numthreads, configInfo.sum_openmp_numthreads );
            }
    	}
    	if ( max_wait_time > 0 ) {
    		printf( "Computing the surface of the Mandelbrot fractal for %lds.\n", max_wait_time );
    	}
    	printf( "\n" );
    }

    // Print detailed info on the current core for processes and threads.
    data_gather_print( &configInfo, option_label, show_numamask, show_rangemask, show_hostname, show_cpu, show_slurmvars );

    // If simulating some load was requested, simulate some load and print the configuration again.

    if ( max_wait_time > 0 ) {

    	double mandel_surface;

    	error = mandelSurface( &configInfo, max_wait_time, &mandel_surface );
    	if ( ( error == 0 ) && ( configInfo.mpi_myrank == 0 ) )
    		printf( "Approximation for the surface of the Mandelbrot fractal: %16.14lg\n\n", mandel_surface );

    	data_gather_print( &configInfo, option_label, show_numamask, show_rangemask, show_hostname, show_cpu, show_slurmvars );

    }

    // Close off properly.

#ifdef WITH_MPI
    MPI_Barrier( MPI_COMM_WORLD );
    MPI_Finalize();
#endif

    return 0;

}

