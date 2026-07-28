#ifndef XT_EVAL_COMPAT_HPP
#define XT_EVAL_COMPAT_HPP

// LOCALDREAM_XT_EVAL(expr): a portability shim for xt::eval().
//
// The xtensor version vendored here (3rdparty/xtensor, the flat
// "xtensor/xarray.hpp" header layout) has a bug in xt::eval(): for an
// expression whose result has FEWER than 4 dimensions it mis-deduces a fixed
// 4-D xtensor<T,4> as the temporary type (it uses xarray's inline shape-buffer
// capacity of 4 as the rank), leaving a garbage 4th extent. That underflows the
// allocation size (a value near SIZE_MAX) and throws std::bad_alloc even with
// gigabytes of memory free. It bites the classifier-free-guidance combine, where
// the batch view drops a dim so the result is 3-D.
//
// TODO: remove this shim once the vendored xtensor is upgraded to a version
// where xt::eval() deduces the result rank correctly; the guard can then go and
// call sites can return to plain xt::eval().

#include <type_traits>
#include <utility>

#include <xtensor/xarray.hpp>

namespace localdream_compat {
template <class E>
inline auto xt_eval(E &&e) {
  using value_type = typename std::decay_t<E>::value_type;
  return ::xt::xarray<value_type>(std::forward<E>(e));
}
}  // namespace localdream_compat

#define LOCALDREAM_XT_EVAL(...) ::localdream_compat::xt_eval(__VA_ARGS__)

#endif  // XT_EVAL_COMPAT_HPP
