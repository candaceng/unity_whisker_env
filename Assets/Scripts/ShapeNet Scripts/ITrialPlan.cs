using System.Collections.Generic;

public interface ITrialPlan
{
    IEnumerable<TrialSpec> GenerateTrials();
}
