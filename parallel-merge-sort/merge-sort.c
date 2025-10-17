#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <direct.h> // for _mkdir on Windows
#include "time.h"
#include "omp.h"

void print_arr(int *a, int n){
  int count = 0;
  for(int i = 0; i < n; i++){
    printf("%d ", a[i]);
    count++;
    if(count == 10){
      printf("\n");
      count = 0;
    }
  }
}

//Ham tao gia tri "gia ngau nhien" song song, nhanh va an toan hon
void pseudo_number_generator(int *a, int n){
  #pragma omp parallel for //Song song hoa duoc vi khong phu thuoc du lieu  
  for(int i = 0; i < n; i++){
    a[i] = (i * 37 + 19) % n; //Cong thuc tinh nhanh va phan phoi deu
  }
}

void merge(int arr[], int left, int mid, int right) {
  int i, j, k;
  int size1 = mid - left + 1;
  int size2 = right - mid;

  int *L = (int*)malloc(size1 * sizeof(int));
  int *R = (int*)malloc(size2 * sizeof(int));

  //Ham for nay khong song song duoc (?)
  for (i = 0; i < size1; i++){
    L[i] = arr[left + i];
  }
   
  for (j = 0; j < size2; j++){
    R[j] = arr[mid + 1 + j];
  }

  i = 0; j = 0; k = left;

  while (i < size1 && j < size2){
    if (L[i] <= R[j]) {
      arr[k++] = L[i++];
    } else {
      arr[k++] = R[j++];
    }
  }

  while (i < size1){
    arr[k++] = L[i++];
  }

  while (j < size2){
    arr[k++] = R[j++];
  }

  free(L);
  free(R);
}

/**
 * \note omp task: Se khong chay code ngay ma se tao ra 1 task rieng -> dua vao hang doi 
 * -> Khi nao co thread ranh thi co the thuc hien no 
 * -> Cach nay giup tan dung CPU tot hon khi co nhiem vu nho
 * \note firstprivate(left, mid): se copy gia tri tai thoi diem task duoc tao ra vao 1 bien cu bo rieng trong task
 */
void merge_sort(int *a, int left, int right){
  if(left < right){
    int mid = (left + right) / 2;

    //Neu mang du nho thi xu ly tuan tu, tranh tao qua nhieu task
    if(right - left <= 10000){
      merge_sort(a, left, mid);
      merge_sort(a, mid + 1, right);
    }else{
      #pragma omp task shared(a) firstprivate(left, mid)
      merge_sort(a, left, mid);

      #pragma omp task shared(a) firstprivate(right, mid)
      merge_sort(a, mid + 1, right);  

      #pragma omp taskwait //Cho ca 2 task xong roi moi merge
    }
    merge(a, left, mid, right);
  }
}

int main(void){
  int num; 
  double tong = 0.0f;
  printf("Nhap so phan tu can sap xep: ");
  scanf("%d", &num);

  int *a = (int*)malloc(sizeof(int) * num);
  if(a == NULL){
    printf("Khong the cap phat bo nho !\n");
    return 1;
  }

  // Ensure output directory exists (works on Windows and POSIX)
  const char *out_dir = "Time_test";
#ifdef _WIN32
  _mkdir(out_dir);
#else
  mkdir(out_dir, 0755);
#endif

  FILE *output_file = fopen("Time_test/e_500mil.csv", "w");
  if(output_file == NULL){
    printf("Khong the mo file de ghi !\n");
    free(a);
    return 1;
  }
  fprintf(output_file, "Lan,thoi_gian(s)\n");

  //Thay doi so core xu ly 
  omp_set_num_threads(8);

  for(int run = 1; run <= 15; run++){
    pseudo_number_generator(a, num);

    double start_time = omp_get_wtime();

    #pragma omp parallel 
    {
      #pragma omp single //Dam bao chi co 1 thread dau tien goi merge_sort(), sau do sinh ra task
      merge_sort(a, 0, num - 1);
    }

    double end_time = omp_get_wtime();

    double elapsed = end_time - start_time;
    tong += elapsed;

  printf("Lan %02d: %.10f\n", run, elapsed);
    fprintf(output_file, "%d,%.10f\n", run, elapsed);
  }

  double aver = tong / 15.0f;
  printf("Thoi gian trung binh: %.10f", aver);
  fprintf(output_file, "Average,%.10f\n", aver);

  fclose(output_file);
  free(a);
  return 0;
}