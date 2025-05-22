# Параллелизация задач финансовой математики на примере формулы Блэка-Шоулса

Данное исследование посвящено разработке оптимальной вычислительной
схемы для расчета цен больших пакетов опционов с участием метода
Монте-Карло.

## Цель

Рассчитать пакет из $100$ опционов, метод Монте-Карло по $10^5$ траекторий.
Исследовать разные способы параллелизации данной задачи.

## Вычислительное окружение
| CPU | Arch | OS | Compiler |
|:---:|:---:|:--:|:--:|
| Apple M3 Pro (11 cores) | arm64v8 | macOS 15.4.1 24E263 | Clang 19.1.7 |
| Intel(R) Core(TM) i5-11400F @ 2.60GHz (6 cores) | x64 | Arch Linux kernel 6.13.8 | Clang 19.1.7 |

## Наивная реализация

Самый очевидный параллелизм здесь - это распределить опционы на несколько
пакетов и рассчитать их на разных потоках. Поскольку полностью
отсутствует зависимость по данным, все достаточно просто - применим LLVM OpenMP 18.1.8.

Функция расчета для одного опциона будет такой:

```cpp
double computeScalar(double S0, double sigma, double r, double T, double K,
                     std::size_t N) {
  double CT = 0;
  std::mt19937_64 rng;

  double sigma2 = sigma * sigma;
  std::lognormal_distribution<double> dist{(r - sigma2 / 2) * T, sigma2 * T};
  double norm = std::exp(-r * T) / N;

  for (std::size_t i = 0; i < N; ++i) {
    double Y = dist(rng);
    double SiT = Y * S0;
    double rem = norm * std::max(0.0, SiT - K);
    CT += rem;
  }

  return CT;
}
```

## Векторизация расчета одного опциона

В предыдущей версии не учтена возможность параллелизации по
данным внутри расчета одного опциона (SIMD).

Чтобы написать переносимый код, хорошо векторизуемый современными компиляторами,
достаточно использовать модель вектора как небольшого массива.

```cpp
double computeVector(double S0, double sigma, double r, double T, double K,
                     std::size_t N) {
  constexpr std::size_t kVectorSize = 8;
  constexpr std::size_t kAlignment = kVectorSize * sizeof(double);
  assert(N % kVectorSize == 0);

  std::array<std::mt19937_64, kVectorSize> rngs;
  std::array<std::lognormal_distribution<double>, kVectorSize> dists;

  double sigma2 = sigma * sigma;
  double norm = std::exp(-r * T) / N;

#pragma unroll
  for (std::size_t lane = 0; lane < kVectorSize; ++lane) {
    rngs[lane].seed(lane);
    dists[lane] =
        std::lognormal_distribution<double>{(r - sigma2 / 2) * T, sigma2 * T};
  }

  alignas(kAlignment) double CTs[kVectorSize]{};

  for (std::size_t i = 0; i < N; i += kVectorSize) {
    alignas(kAlignment) double Ys[kVectorSize];

#pragma omp simd
    for (std::size_t lane = 0; lane < kVectorSize; ++lane) {
      Ys[lane] = dists[lane](rngs[lane]);
    }

#pragma omp simd
    for (std::size_t lane = 0; lane < kVectorSize; ++lane) {
      double SiT = Ys[lane] * S0;
      double rem = norm * std::max(0.0, SiT - K);
      CTs[lane] += rem;
    }
  }

  return std::accumulate(std::begin(CTs), std::end(CTs), 0.0);
}
```

Автовекторизация логнормального распределения оказалась все же невозможной,
оптимизатор не справился.

Однако, если произвести замеры, то окажется, что выигрыш по сравнению с
наивной версией все же есть. Результаты бенчмарка в [таблице](report/results.csv).
Векторизованная реализация отработала за $14.7\pm0.2$ мс, наивная
за $19.7\pm0.6$ мс. Ускорение в 1.3 раза. Здесь наибольший вклад
внес вероятно unroll цикла с генерацией значений $Y$, также имела место
векторизация вычисления суммы. При замерах фиксировалось число потоков
=11.

Попробуем на второй машине (указана во второй строке таблицы в пункте окружение).
Результаты в [другой](report/results_x64.csv) таблице. Замеры производились без гипертрединга,
то есть число потоков =6. Результаты схожи. Анализ ассемблера можно найти по [ссылке](https://godbolt.org/z/vTfEns3Tc).
Видно, что действительно последний цикл успешно векторизовался, а `operator()` из `std::lognormal_distribution` даже не
заинлайнился.

## Вывод

Возможности оптимизаторов в области параллелизма все еще не безграничны,
часто требуется ручное вмешательство.
