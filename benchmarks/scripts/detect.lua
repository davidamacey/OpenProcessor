-- wrk script for benchmarking /detect endpoint
-- Usage: wrk -t4 -c64 -d30s -s benchmarks/scripts/detect.lua http://localhost:4603/detect

local counter = 0
local threads = {}

-- Read test image on setup
function setup(thread)
   thread:set("id", counter)
   table.insert(threads, thread)
   counter = counter + 1
end

function init(args)
   -- Read test image file
   local file = io.open("test_images/sample.jpg", "rb")
   if file then
      image_data = file:read("*all")
      file:close()
   else
      print("Warning: Could not read test_images/sample.jpg")
      image_data = ""
   end

   -- Build multipart form data
   local boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"

   body = "--" .. boundary .. "\r\n"
   body = body .. 'Content-Disposition: form-data; name="image"; filename="test.jpg"\r\n'
   body = body .. "Content-Type: image/jpeg\r\n\r\n"
   body = body .. image_data
   body = body .. "\r\n--" .. boundary .. "--\r\n"

   wrk.method = "POST"
   wrk.body = body
   wrk.headers["Content-Type"] = "multipart/form-data; boundary=" .. boundary
end

function request()
   return wrk.format()
end

function response(status, headers, body)
   if status ~= 200 then
      print("Error: " .. status)
   end
end

function done(summary, latency, requests)
   io.write("------------------------------\n")
   io.write(string.format("Requests/sec: %.2f\n", requests.rate))
   io.write(string.format("Transfer/sec: %.2fMB\n", requests.rate * #body / 1024 / 1024))
   io.write(string.format("Avg Latency:  %.2fms\n", latency.mean / 1000))
   io.write(string.format("Max Latency:  %.2fms\n", latency.max / 1000))
   io.write(string.format("Stdev:        %.2fms\n", latency.stdev / 1000))
   io.write(string.format("Percentiles:\n"))
   for _, p in pairs({ 50, 90, 99, 99.9 }) do
      n = latency:percentile(p)
      io.write(string.format("  %g%%: %.2fms\n", p, n / 1000))
   end
end
