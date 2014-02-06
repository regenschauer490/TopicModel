/*
 * Copyright (c) 2013, Daniel Park
 * All rights reserved.
 *
 * Permission to modify and redistribute this software is granted to
 * anyone provided the above copyright notice, this condition and the
 * following disclaimer are retained.
 *
 * This software is provided "as is", without and express or implied
 * warranty. In no event shall the author be liable for damages arising
 * from the use of this software.
 */

#pragma once

#include <deque>
#include <list>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "eval.h"

template <class C>
struct container_traits
{
    // Type value_type
    // void add_element(C&,T)
    // void concat(C&,C)
    // Type rebind<U>
};

template<class C>
struct sequence_container_traits;

template<template<class,class> class C, class T, class A>
struct sequence_container_traits<C<T,A>>
{
    using value_type = T;

    static void add_element(C<T,A>& c, const T& t)
    {
        c.push_back(t);
    }

    static void concat(C<T,A>& lhs, const C<T,A>& rhs)
    {
        lhs.insert(lhs.end(),rhs.begin(),rhs.end());
    }

    template<class U>
    using rebind = C<U,typename A::template rebind<U>::other>;
};

template<class... Args>
struct container_traits<std::deque<Args...>> : public sequence_container_traits<std::deque<Args...>>
{};

template<class... Args>
struct container_traits<std::list<Args...>> : public sequence_container_traits<std::list<Args...>>
{};

template<class... Args>
struct container_traits<std::vector<Args...>> : public sequence_container_traits<std::vector<Args...>>
{};

template<class C>
struct associative_container_traits;

template<template<class,class,class> class C, class T, template<class> class O, class A>
struct associative_container_traits<C<T,O<T>,A>>
{
    using value_type = T;

    static void add_element(C<T,O<T>,A>& c, const T& t)
    {
        c.insert(t);
    }

    static void concat(C<T,O<T>,A>& lhs, const C<T,O<T>,A>& rhs)
    {
        lhs.insert(rhs.begin(),rhs.end());
    }

    template<class U>
    using rebind = C<U,O<U>,typename A::template rebind<U>::other>;
};

template<class... Args>
struct container_traits<std::multiset<Args...>> : public associative_container_traits<std::multiset<Args...>>
{};

template<class... Args>
struct container_traits<std::set<Args...>> : public associative_container_traits<std::set<Args...>>
{};

template<class C>
struct hash_container_traits;

template<template<class,class,class,class> class C, class T, template<class> class H, template<class> class O, class A>
struct hash_container_traits<C<T,H<T>,O<T>,A>>
{
    using value_type = T;

    static void add_element(C<T,H<T>,O<T>,A>& c, const T& t)
    {
        c.insert(t);
    }

    static void concat(C<T,H<T>,O<T>,A>& lhs, const C<T,H<T>,O<T>,A>& rhs)
    {
        lhs.insert(rhs.begin(),rhs.end());
    }

    template<class U>
    using rebind = C<U,H<U>,O<U>,typename A::template rebind<U>::other>;
};

template<class... Args>
struct container_traits<std::unordered_multiset<Args...>> : public hash_container_traits<std::unordered_multiset<Args...>>
{};

template<class... Args>
struct container_traits<std::unordered_set<Args...>> : public hash_container_traits<std::unordered_set<Args...>>
{};

// basic_string
template<class T, template<class> class K, class A>
struct container_traits<std::basic_string<T,K<T>,A>>
{
    using value_type = T;

    static void add_element(std::basic_string<T,K<T>,A>& c, const T& t)
    {
        c.push_back(t);
    }

    static void concat(std::basic_string<T,K<T>,A>& lhs, const std::basic_string<T,K<T>,A>& rhs)
    {
        lhs+=rhs;
    }

    template<class U>
    using rebind = std::basic_string<U,K<U>,typename A::template rebind<U>::other>;
};
