# SCOPE Script Learning Notes

## Read/Write

### Flat data files

``` sql
data = EXTRACT
        id: int,
        name: string
    FROM
        "/data/sample_input.tsv"
    USING DefaultTextExtractor();
    // Read csv, silent, skip header
    // USING DefaultTextExtractor(delimiter: ',', silent: true, skipFirstNRows: 1);

OUTPUT data
    TO "/data/output.tsv"
    USING DefaultTextOutputter();
    // Write csv, escape, with header
    // USING DefaultTextOutputter(delimiter: ',', escape: true, outputHeader: true);
```

### Stream

``` sql
ss_14 = SELECT *
    FROM SSTREAM "AssociationsSlice_2019-04-14.ss";

OUTPUT MonetizationPADistinctData
    TO SSTREAM @Out_MonetizationPAData
    CLUSTERED BY RGUID
    SORTED BY RGUID
    WITH STREAMEXPIRY "90";
```

## LINQ and lambdas in expressions

LINQ expression are built in as columns selector, but notice that list need to be serialized to output streams.

``` sql
rs1 = SELECT
    Urls.Split(';').Where(u => u.StartsWith("https:")).ToList() AS HttpUrls
  FROM searchlog;
rs2 = SELECT
    string.Join(";", HttpUrls) AS HttpUrls
  FROM rs1;
```

## Choose correct short circuit grammar

If you'd like to use short circut null detection, beware to use `&&` not `AND`, since later expression could cause NullReference crush on cluster because of Scope optimization.

## JOIN

CROSS JOIN is basically an INNER JOIN without a join condition – it returns all the possible combinations of records. This is the Cartesian product of the records in both rowsets.

Do not use implicit JOIN. In some SQL variants – but not Scope - the WHERE condition is optional. When it doesn't appear, this implicit join behaves as a CROSS JOIN. As you can see, it's easy to accidently do an (expensive) CROSS JOIN when using such a syntax. It's for this reason Implicit Join should be avoided even in languages that support it.

``` sql
// DO NOT
rs0 =
  SELECT
    employees.DepartmentID AS EmpDepartmentId,
    departments.DepartmentID AS DepDepartmentID,
    employees.LastName,
    departments.DepartmentName
  FROM employees , departments
  WHERE employees.DepartmentID == departments.DepartmentID;

// DO
rs0 =
  SELECT
    employees.DepID AS EmpDepId,
    departments.DepID,
    employees.EmpName,
    departments.DepName
  FROM employees
    LEFT OUTER JOIN departments ON employees.DepID == departments.DepID;

```

Reference:
<http://en.wikipedia.org/wiki/Join_(SQL)>

## Group and Aggregation

``` sql
rs1 = SELECT
    ROW_NUMBER,
    Region,
    COUNT(DISTINCT SessionId) AS NumSessions,
    COUNTIF(Duration > 600) AS NumLongSessions,
    SUM(Duration) AS TotalDuration,
    ARGMAX(Duration, Ip) AS MaxDurationIp  // find the session ip with max duration
  FROM searchlog
  GROUP BY Region;
```

### Window functions

Aggregate functions: An aggregator will compute a single result value over a group of values and will have an identity value for the case that the group is empty. They include reporting aggregate functions such as SUM or AVG, cumulative aggregate functions, and moving aggregate functions.

### Native supported aggregates

* ANY_VALUE
* ARGMAX(a, b)
* AVG
* COUNT
* COUNTIF
* FIRST
* LAST
* LIST
* MAX
* MIN
* STDEV
* STDEVP
* SUM
* VAR
* VARP

``` sql
rs1 =
  SELECT
    Name,
    ProductCategoryId,
    SUM(ListPrice) OVER(PARTITION BY ProductCategoryId) AS PricebyCategory
  FROM product;
```

### Ranking functions

Returns a ranking value for each row in a partition. Examples include DENSE_RANK, ROW_NUMBER, NTILE, and RANK. There is no guarantee that rows will be returned in the same order in consecutive executions (e.g. these are non-deterministic).

``` sql
rs2 =
  SELECT
    Name, ProductCategoryId, ListPrice,
    ROW_NUMBER() OVER (PARTITION BY ProductCategoryId ORDER BY ListPrice) AS RowNumber,
    RANK() OVER (PARTITION BY ProductCategoryId ORDER BY ListPrice) AS Rank,
    DENSE_RANK() OVER (PARTITION BY ProductCategoryId ORDER BY ListPrice) AS DenseRank
  FROM product;
```

### Percentile functions

Analytic functions compute an aggregate value based on a group of rows. However, unlike aggregate functions, they can return multiple rows for each group. Examples include cumulative distribution, or functions that access data from a previous row in the same result set without the use of a self-join.

``` sql
rs3 =
  SELECT Name, ProductCategoryId, ListPrice,
    CUME_DIST() OVER (PARTITION BY ProductCategoryId ORDER BY ListPrice) AS CumeDist
  FROM product;
```

## Pivot / Unpivot

PIVOT transforms the rowset rows into columns. PIVOT rotates the unique values in one column into multiple columns in the output, and performs aggregations on remaining column values, as needed.

UNPIVOT does the opposite operation to transform a set of rowset columns into rows.

``` sql
data =
  SELECT *
  FROM (VALUES
    ("a1", "b1", "c1", "d1"),
    ("a2", "b2", "c2", "d2"),
    ("a3", "b3", "c3", "d3"),
    ("a4", "b4", "c4", "d4"))
  AS Source(a, b, c, d);

// pivot data
Rs1_Pivot =
  SELECT *
  FROM data
  PIVOT (ANY_VALUE(c) FOR d IN (
    "d1" AS d1,
    "d2" AS d2,
    "d3" AS d3,
    "d4" AS d4))
  AS PivotedTable;

Rs2_unpivot =
  SELECT a, b, c, d
  FROM Rs1_Pivot
    UNPIVOT (c FOR d IN (
    d1,
    d2,
    d3,
    d4))
  AS UnpivotedTable;
```

<https://mscosmos.visualstudio.com/CosmosWiki/_wiki/wikis/Cosmos.wiki/585/PIVOT-and-UNPIVOT>

## CROSS APPLY, LIST, and ARRAY_AGG

rs1 looks like

Region|Names
-|-
en-us|A,B
en-GB|C,D

``` sql
rs1 = SELECT
    Region,
    Name
  FROM searchlog;

rs2 = SELECT
    Region,
    SplitNames AS Name
  FROM rs1
  CROSS APPLY Names.Split(',') AS SplitNames;
```

rs2 looks like

Region|Name
-|-
en-us|A
en-us|B
en-GB|C
en-GB|D

``` sql
rs3 = SELECT
    Region,
    String.Join(",", LIST(Name).ToArray()) AS Names
  FROM rs2
  GROUP BY Region;

// more prefered way is ARRAY_AGG
rs3 = SELECT
    Region,
    String.Join(",", ARRAY_AGG(Name)) AS Names
  FROM rs2
  GROUP BY Region;
```

rs3 looks same as rs1

## Custom Extractors

build your own extractors with C# inheriting from base class `Extractor` and implement `Extract`.

``` cs
public class MyTsvExtractor : Extractor
{
    public override Schema Produces(string[] requested_columns, string[] args)
    {
        return new Schema(requested_columns);
    }

    public override IEnumerable<Row> Extract(StreamReader reader, Row output_row, string[] args)
    {
        char delimiter = '\t';
        string line;
        while ((line = reader.ReadLine()) != null)
        {
            var tokens = line.Split(delimiter);
            for (int i = 0; i < tokens.Length; ++i)
            {
                output_row[i].UnsafeSet(tokens[i]);
            }
            yield return output_row;
        }
    }
}
```

``` sql
// call the custom extractor
searchlog =
    EXTRACT IId:int, UId:int, Start:DateTime, Market:string, Query:string, DwellTime:int, Results:string, ClickedUrls:string
    FROM @In_SearchLog
    USING MyTsvExtractor();
```

### User-Defined Types (UDTs)

By defining your own type, with constructor as a deserializer, and implement an explicit `Serialize` method, then you canuse the UDT in scope query like built-in types.

You should override Equals() and GetHashCode() in any UDT you write. Here is some guidance on approaches you can use: <http://stackoverflow.com/questions/263400/what-is-the-best-algorithm-for-an-overridden-system-object-gethashcode>.

``` cs
public class UrlList
{
    public List<string> Items;
    private static char [] sepchars = new char[] {';'};

    public UrlList(string s)
    {
        this.Items = new List<string>(s.Split(sepchars));
    }

    public static UrlList Create(string s)
    {
        return new UrlList(s);
    }

    public string Serialize()
    {
        return string.Join(";", this.Items);
    }
}
```

``` sql
searchlog =
  EXTRACT IId:int, UId:int, Start:DateTime, Market:string, Query:string, DwellTime:int, Results:string, ClickedUrls:string
  FROM @"SampleInputs\SearchLog.txt"
  USING DefaultTextExtractor();

searchlog2 =
  SELECT IId, UId, Start, Market, Query, DwellTime, Results, new UrlList (ClickedUrls) AS CLickedUrlsList
  FROM searchlog;

searchlog3 =
  SELECT IId, UId, Start, Market, Query, DwellTime, Results, CLickedUrlsList.Serialize() AS ClickedUrls
  FROM searchlog2;

OUTPUT searchlog3 TO @"D:\output-searchlog.txt";
```

## Processor

A processor allows you to programmatically transform a rowset. Processors can modify the values of a rowset, add columns, remove columns, remove rows, and create new rows.

``` cs
public class MyProcessor : Processor
{
    public override Schema Produces(string[] requested_columns, string[] args, Schema input_schema)
    {
        var output_schema = input_schema.Clone();
        var newcol = new ColumnInfo("Market2", typeof(string));
        output_schema.Add(newcol);
        return output_schema;
    }

    public override IEnumerable<Row> Process(RowSet input_rowset, Row output_row, string[] args)
    {
        foreach (Row input_row in input_rowset.Rows)
        {
            input_row.CopyTo(output_row);
            string market = input_row[0].String;
            output_row[2].Set( "FOO" + market );
            yield return output_row;
        }
    }
}
```

``` sql
rs1 = SELECT Market, Results
      FROM searchlog;

rs2 = PROCESS rs1
      PRODUCE Market, Results, Market2
      USING MyProcessor;
```

## Debug Stream

``` cs
Using ScopeRuntime.Diagnostics;

public class MyTsvExtractor : Extractor
{
    public override Schema Produces(string[] requested_columns, string[] args)
    {
        return new Schema(requested_columns);
    }

    public override IEnumerable<Row> Extract(StreamReader reader, Row output_row, string[] args)
    {
        while ((line = reader.ReadLine()) != null)
        {
            try { … }
            catch (SomeException exc)
            {
                ScopeRuntime.Diagnostics.DebugStream.WriteLine(exc.Message);
            }
            yield return output_row;
        }
    }

}
```

Then when submit the job, open Job Properties and set "Debug Stream Path". After job done, you can find the exceptions log in the debug stream.

you can set break points (F9) in a script, C# behind file, and reference C# project to do step-by-step debugging (F10), as well as step into the methods called in the statement(F11). Also, you can add a RowSet name to the watch window and preview the content with visualizer.

If you have trouble using F11 to step into methods, please make sure that "Enable Just My Code" has been selected in Visual Studio > Debug > Options and Settings.

If the output preview window is not opened automatically, you can click the Local Stream Preview button under the Scope menu to open it.

## Optimization 101

Here are some common ways to help increase performance of SCOPE jobs:

* Avoid outputting intermediate results in production scripts
* Break independent commands into separate scripts
* Avoid data skew and low distinctness bottlenecks
* Always use CLUSTERED BY for structured streams
* Use system built-in operators before User-Defined Operators

## Azure Data Lake with Cosmos

### Install PowerShell Core

<https://github.com/PowerShell/PowerShell>

### Install Az PowerShell

``` powershell
Install-Module -Name Az -AllowClobber -Force
Get-InstalledModule -Name Az -AllVersions | Select-Object Name,Version
Disable-AzContextAutoSave

$alias = "TBD"
$appId = "TBD"
$password = "TBD"
$scopeAdlExePath = "c:\local\sandbox\ScopeAdlCommand\"

# Subscription and Account for Connection to ADLA/ADLS
$subscription = "Cosmos_C&E_DPG_BigData_100036"
$tenant="72f988bf-86f1-41af-91ab-2d7cd011db47"
$account="sandbox-c08"

# Connect to ADLS
$accountSuffix = ".azuredatalakestore.net"
$pscredential = New-Object System.Management.Automation.PSCredential `
  ($appId,(ConvertTo-SecureString$password-AsPlainText-Force))
Connect-AzAccount -ServicePrincipal -Tenant $tenant -Credential $pscredential
```

### Install ScopeADL.exe

<http://aka.ms/ScopeADLexe>
unzip files to c:\local\sandbox\ScopeAdlCommand

### Mapping VC names to ADL account names

sandbox on Cosmos08 -> sandbox-c08

Team's VC names mapps to ADL `bingads-bi-oi-c08`, Azure link:
<http://aka.ms/azureportalprod>

<https://portal.azure.com/?feature.customPortal=false#@microsoft.onmicrosoft.com/resource/subscriptions/121ed073-2274-4c26-8287-896e4afef998/resourceGroups/conv/providers/Microsoft.DataLakeAnalytics/accounts/bingads-bi-oi-c08/overview>

In Cosmos `users` and `my` are special redirects, in ADLS they are just normal folder.

Prefered data formats:

* TSV
* SS
* Parquet
* JSON

Parquet reader/writer

``` cs
Extractors.Parquet();
Outputter.Parquet();
```

### Access ADLS in Scope

Menu Tools -> Data Lake -> Open ADLS Path

adl://sandbox-c08.azuredatalakestore.net/local/labs/

In scope script, accessing stream like:

``` scope
#DECLARE outputStream string = "/local/labs/input-origin.tsv";
```

1. Right click the script and select "Submit Script";
2. In Targeted VC, input "sandbox-c08";
3. Set Parameters if needed;
4. Submit job;

On Azure Data Lake Analytics portal, jobs, find the job, you could see that the input parameters are replaced as the set value, and the stream path is replaced to be like:

``` scope
#DECLARE outputStream string = "adl://sandbox-c08.azuredatalakestore.net/local/labs/input-origin.tsv";
```

### Spark on Cosmos (Azure Synapse) <- Azure Databricks <- Aparche Spark

In memory, fast, open-source, extensible for Scala, Java and Python, Spark SQL, GraphX, Streaming and Machine Learning Library (MLlib).

PySpark: Python library for Spark, similar dataframe operator functions.

Lab Master supports multiple lanugages online interactive console, like Jupyter.
