// Copyright (c) 2009-2013 Craig Henderson
// https://github.com/cdmh/mapreduce

/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#pragma once

namespace mapreduce {
extern void* enabler;

template<typename T> uintmax_t    const length(T const &str);
template<typename T> char const * const data(T const &str);

template<>
inline uintmax_t const length(std::string const &str)
{
    return str.length();
}

template<>
inline char const * const data(std::string const &str)
{
    return str.data();
}

template<typename MapKey, typename MapValue>
class map_task
{
  public:
    typedef MapKey   key_type;
    typedef MapValue value_type;             
};

template<typename ReduceKey, typename ReduceValue>
class reduce_task
{
  public:
    typedef ReduceKey   key_type;
    typedef ReduceValue value_type;
};

template<typename MapTask,
         typename ReduceTask,
         typename Combiner          = null_combiner,
         typename Datasource        = datasource::directory_iterator<MapTask>,
         typename IntermediateStore = intermediates::in_memory<MapTask, ReduceTask>,
         typename StoreResult       = typename IntermediateStore::store_result_type>
class job : detail::noncopyable
{
  public:
    typedef MapTask           map_task_type;
    typedef ReduceTask        reduce_task_type;
    typedef Combiner          combiner_type;
    typedef Datasource        datasource_type;
    typedef IntermediateStore intermediate_store_type;

    typedef
    typename intermediate_store_type::const_result_iterator
    const_result_iterator;

    typedef
    typename intermediate_store_type::keyvalue_t
    keyvalue_t;

	using make_key_t = typename std::conditional<
		std::is_default_constructible<typename map_task_type::key_type>::value,
		typename map_task_type::key_type,
		std::unique_ptr<typename map_task_type::key_type>
	>::type;

  private:
    class map_task_runner : detail::noncopyable
    {
      public:
        typedef ReduceTask reduce_task_type;

        map_task_runner(job &j)
          : job_(j),
            intermediate_store_(job_.number_of_partitions())
        {
        }

        // 'value' parameter is not a reference to const to enable streams to be passed
        map_task_runner &operator()(typename map_task_type::key_type const &key,
                                    typename map_task_type::value_type     &value)
        {
            map_task_type()(*this, key, value);

            // consolidating map intermediate results can save time by
            // aggregating the mapped valued at mapper
            combiner_type instance;
            intermediate_store_.combine(instance);

            return *this;
        }

        template<typename T>
        bool const emit_intermediate(T const &key, typename reduce_task_type::value_type const &value)
        {
            return intermediate_store_.insert(key, value);
        }

        intermediate_store_type &intermediate_store(void)
        {
            return intermediate_store_;
        }

      private:
        job                     &job_;
        intermediate_store_type  intermediate_store_;
    };

    class reduce_task_runner : detail::noncopyable
    {
      public:
        reduce_task_runner(
            std::string					output_filespec,
            unsigned					partition,
            unsigned					num_partitions,
            intermediate_store_type&	intermediate_store,
            results&					result)
          : partition_(partition),
            result_(result),
            intermediate_store_(intermediate_store),
            store_result_(output_filespec, partition, num_partitions)
        {
        }

        void reduce(void)
        {
            intermediate_store_.reduce(partition_, *this);
        }

        void emit(typename reduce_task_type::key_type   const &key,
                  typename reduce_task_type::value_type const &value)
        {
            intermediate_store_.insert(key, value, store_result_);
        }

        template<typename It>
        void operator()(typename reduce_task_type::key_type const &key, It it, It ite)
        {
            ++result_.counters.reduce_keys_executed;
            reduce_task_type()(*this, key, it, ite);
            ++result_.counters.reduce_keys_completed;
        }

      private:
        unsigned					partition_;
        results&					result_;
        intermediate_store_type&	intermediate_store_;
        StoreResult					store_result_;
    };

  public:
    job(datasource_type &datasource, specification const &spec)
      : datasource_(datasource),
        specification_(spec),
        intermediate_store_(specification_.reduce_tasks)
     {
     }

    const_result_iterator begin_results(void) const
    {
        return intermediate_store_.begin_results();
    }

    const_result_iterator end_results(void) const
    {
        return intermediate_store_.end_results();
    }

	template<class K, typename std::enable_if<std::is_same<K, std::unique_ptr<typename map_task_type::key_type>>::value>::type*& = enabler>
	bool const get_next_map_key(K &key)
    {
		return datasource_.setup_key_new(key);
    }

	template<class K, typename std::enable_if<std::is_same<K, typename map_task_type::key_type>::value>::type*& = enabler>
	bool const get_next_map_key(K &key)
	{
		return datasource_.setup_key(key);
	}

    unsigned const number_of_partitions(void) const
    {
        return specification_.reduce_tasks;
    }

    unsigned const number_of_map_tasks(void) const
    {
        return specification_.map_tasks;
    }

    template<typename SchedulePolicy>
    void run(results &result)
    {
        SchedulePolicy schedule;
        run(schedule, result);
    }

    template<typename SchedulePolicy>
    void run(SchedulePolicy &schedule, results &result)
    {
		reset_result();
        auto const start_time = std::chrono::system_clock::now();
        schedule(*this, result);
        result.job_runtime = std::chrono::system_clock::now() - start_time;
    }

	template<typename Sync>
	bool const run_map_task(std::unique_ptr<typename map_task_type::key_type> &key, results &result, Sync &sync)
	{
		return run_map_task_impl(*key, result, sync);
	}

	template<typename Sync>
	bool const run_map_task(typename map_task_type::key_type &key, results &result, Sync &sync)
	{
		return run_map_task_impl(key, result, sync);
	}

    template<typename Sync>
    bool const run_map_task_impl(typename map_task_type::key_type &key, results &result, Sync &sync)
    {
        auto const start_time = std::chrono::system_clock::now();

        try
        {
            ++result.counters.map_keys_executed;

            // get some data
            typename map_task_type::value_type value;
            if (!datasource_.get_data(key, value))
            {
                ++result.counters.map_key_errors;
                return false;
            }

            map_task_runner runner(*this);
            runner(key, value);

            // merge the map task intermediate results into the job
            std::lock_guard<Sync> lock(sync);
            intermediate_store_.merge_from(runner.intermediate_store());
            ++result.counters.map_keys_completed;
        }
        catch (std::exception &e)
        {
            std::cerr << "\nError: " << e.what() << "\n";
            ++result.counters.map_key_errors;
            return false;
        }
        result.map_times.push_back(std::chrono::system_clock::now() - start_time);

        return true;
    }

    void run_intermediate_results_shuffle(unsigned const partition)
    {
        intermediate_store_.run_intermediate_results_shuffle(partition);
    }

    bool const run_reduce_task(unsigned const partition, results &result)
    {
        bool success = true;

        auto const start_time(std::chrono::system_clock::now());
        try
        {
            reduce_task_runner runner(
                specification_.output_filespec,
                partition,
                number_of_partitions(),
                intermediate_store_,
                result);
            runner.reduce();
        }
        catch (std::exception &e)
        {
            std::cerr << "\nError: " << e.what() << "\n";
            ++result.counters.reduce_key_errors;
            success = false;
        }
        
        result.reduce_times.push_back(std::chrono::system_clock::now() - start_time);

        return success;
    }

	void reset_result()
	{
		intermediate_store_.reset_result();
	}

  private:
    datasource_type			datasource_;
    specification			specification_;
    intermediate_store_type	intermediate_store_;
};

}   // namespace mapreduce

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
