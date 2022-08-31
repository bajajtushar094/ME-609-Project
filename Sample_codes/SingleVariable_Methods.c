#include<stdio.h>
# include <stdlib.h>
# include <math.h>
#include <string.h>

double Exhaustive_Search(double a, double b);
double Golden_Section_Search (double a, double b);
double objective_function(double x);
double xval(double w, double a, double b);

int main()
{	
	double a, b; /*Ranges of x*/
	printf("Enter\n a = lower limit of x\n b = upper limit on x\n");
	scanf("%lf %lf",&a,&b);
	
	Exhaustive_Search(a,b);
	/* Bounding_Phase(); */
	/* Golden_Section_Search(a,b); */
	return(0);
}

double objective_function(double x)
{
	return(x*x + (54/x));
}

double Exhaustive_Search(double a, double b)
{
	int i, n, feval;
	double delta;
	double x1, x2, x3, f1, f2, f3;
	FILE *out;
	
	printf("\n**********************************\n");
	printf("Exhaustive Search Method\n");
	printf("Enter 'n' = number of steps\t");
	scanf("%d",&n);
	/*Step 1*/
	delta = (b-a)/(double) n; /*Set delta value*/
	x1 = a; x2 = x1 + delta; x3 = x2 + delta; /*New points*/
	feval = 0; /*function evaluation*/
	f1 = objective_function(x1); /*Calculate objective_function*/
	f2 = objective_function(x2);
	f3 = objective_function(x3);
	i = 1; /*Number of iterations*/
	feval = feval + 3;
	out = fopen("Exhaustive_Search_iterations.out","w");/*Output file*/
	fprintf(out,"#It\tx1\tx2\tx3\tf(x1)\tf(x2)\tf(x3)\n");
	do{
		fprintf(out,"%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",i,x1,x2,x3,f1,f2,f3);
		/*Step 2: Termination condition */
		if(f1 >= f2){
			if(f2 <= f3){
				break;
			}
		}
		/*If not terminated, update values*/
		x1 = x2; x2 = x3; x3 = x2 + delta;
		f1 = f2;
		f2 = f3;
		f3 = objective_function(x3);
		feval++;
		i = i + 1;
	}while(x3 <= b);/*Step 3*/
	printf("\n**********************************\n");
	printf("The minimum point lies between (%lf,%lf)",x1,x3);
	printf("\n#Total number of function evaluations: %d",feval);
	/*Store in the file*/
	fprintf(out,"\n#The minimum point lies between (%lf,%lf)",x1,x3);
	fprintf(out,"\n#Total number of function evaluations: %d",feval);
	fclose(out);
}

double Golden_Section_Search (double a, double b)
{
	int k, feval;
	double w1, w2, a_w, b_w, L_w, f1, f2;
	double epsilon, golden_no;
	FILE *out;
	
	printf("\n**********************************\n");
	printf("Golden Section Search Method\n");
	printf("Enter 'epsilon' = \t");
	scanf("%lf",&epsilon);
	feval = 0; /*function evaluation*/
	/*Step 1*/
	/*Step 2*/
	/*Step 3*/

	return(0);
}

double omega(double x, double a, double b)
{
	if(b <= a){
		printf("\nProblem in passing the limits");
		exit(0);
	}
	return((x-a)/(b-a));
}

double xval(double w, double a, double b)/*for Golden_Section_Search*/
{
	if(b <= a){
		printf("\nProblem in passing the limits");
		exit(0);
	}
	return(a + w*(b-a));
}